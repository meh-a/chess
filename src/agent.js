'use strict';

// ── Dirichlet noise helpers ───────────────────────────────────────────────────

function _randn() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function _gammaRandom(a) {
    if (a < 1) return _gammaRandom(a + 1) * Math.pow(Math.random(), 1 / a);
    const d = a - 1/3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
        let x = _randn(), v = 1 + c * x;
        while (v <= 0) { x = _randn(); v = 1 + c * x; }
        v = v * v * v;
        const u = Math.random();
        if (u < 1 - 0.331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
}
function _dirichletSample(alpha, k) {
    const s = new Float32Array(k);
    let sum = 0;
    for (let i = 0; i < k; i++) { s[i] = _gammaRandom(alpha); sum += s[i]; }
    for (let i = 0; i < k; i++) s[i] /= sum;
    return s;
}

// Compute material balance from an encoded position vector (first 768 values).
// White planes 0-5, black planes 6-11; piece order: pawn,knight,bishop,rook,queen,king.
// Returns a value in [-1, 1] (same scale as the value head).
function _matFromPos(pos) {
    const vals = [1, 3, 3, 5, 9, 0];
    let score = 0;
    for (let p = 0; p < 6; p++) {
        const v = vals[p];
        const wBase = p * 64, bBase = (p + 6) * 64;
        for (let s = 0; s < 64; s++) {
            if (pos[wBase + s]) score += v;
            if (pos[bBase + s]) score -= v;
        }
    }
    return score / 39;
}

// ── MCTSNode ──────────────────────────────────────────────────────────────────

class MCTSNode {
    constructor(prior = 1.0) {
        this.prior     = prior;   // P(s,a) – policy prior from parent's NN call
        this.visits    = 0;       // N(s,a)
        this.valueSum  = 0;       // W(s,a) – cumulative value from WHITE's perspective
        this.children  = null;    // null until expanded; Map<moveKey, {move, node}>
    }
    get q() { return this.visits > 0 ? this.valueSum / this.visits : 0; }
}

// ── MCTS ──────────────────────────────────────────────────────────────────────

class MCTS {
    // sims      – number of simulations per search
    // cPuct     – exploration constant (higher = more exploration)
    // matWeight – blend material score into leaf evaluation (0 = pure NN,
    //             0.3 = 30% material + 70% NN). Useful early in training when
    //             the value head outputs near-zero for all positions.
    constructor(nn, sims = 50, cPuct = 1.5, matWeight = 0) {
        this.nn        = nn;
        this.sims      = sims;
        this.cPuct     = cPuct;
        this.matWeight = matWeight;
        this._root     = null; // retained between moves for tree reuse
    }

    // Advance the retained root to the child reached by `move`, discarding the
    // rest of the tree.  Call this after every move during self-play.
    advanceRoot(move) {
        if (!this._root || !this._root.children) { this._root = null; return; }
        const entry = this._root.children.get(move.from * 64 + move.to);
        this._root = entry ? entry.node : null;
    }

    // Run simulations from rootGame and return the root MCTSNode.
    //
    // training=true  – enables Dirichlet noise on root priors (exploration)
    //                  and activates tree reuse via this._root.
    //
    // Uses makeMove/undoMove instead of cloning, and batches NN evaluations
    // across sims to minimise GPU→CPU syncs.
    search(rootGame, training = false) {
        const root = (training && this._root) ? this._root : new MCTSNode(1.0);
        if (training) this._root = root;

        const EVAL_BATCH = 32;
        let noiseApplied = !training; // will be set true after root is first expanded

        // If root already has children (tree reuse), apply noise immediately.
        if (training && root.children !== null && root.children.size > 0) {
            this._addRootNoise(root);
            noiseApplied = true;
        }

        for (let simStart = 0; simStart < this.sims; simStart += EVAL_BATCH) {
            const count = Math.min(EVAL_BATCH, this.sims - simStart);

            const allPaths    = [];
            const allLeaves   = [];
            const toEval      = [];
            const nodeEvalIdx = new Map();

            // ── Selection ─────────────────────────────────────────────────
            for (let s = 0; s < count; s++) {
                let node = root;
                const path = [node];
                const undos = [];

                try {
                    while (node.children !== null) {
                        const entry = this._selectChild(node, rootGame.turn);
                        if (!entry) break;
                        undos.push(rootGame.makeMove(entry.move));
                        node = entry.node;
                        path.push(node);
                    }

                    const status = rootGame.status();
                    allLeaves.push({ node, status });

                    if (status === 'playing' && node.children === null && !nodeEvalIdx.has(node)) {
                        nodeEvalIdx.set(node, toEval.length);
                        toEval.push({
                            node,
                            pos:   rootGame.encodePosition(),
                            moves: rootGame.legalMoves().filter(m => !m.promo || Math.abs(m.promo) === 5),
                            value: 0,
                        });
                    }
                } finally {
                    for (let i = undos.length - 1; i >= 0; i--) rootGame.undoMove(undos[i]);
                }

                allPaths.push(path);
            }

            // ── Batch NN evaluation + expansion ───────────────────────────
            if (toEval.length > 0) {
                const results = this.nn.predictBatch(toEval.map(e => e.pos));

                for (let j = 0; j < toEval.length; j++) {
                    const entry         = toEval[j];
                    const { v, policy } = results[j];
                    entry.value = this.matWeight > 0
                        ? (1 - this.matWeight) * v + this.matWeight * _matFromPos(entry.pos)
                        : v;

                    // policy = [POLICY_K×from-embeddings (64*POLICY_K), POLICY_K×to-embeddings (64*POLICY_K)]
                    // score(m) = dot(fromEmbed[m.from], toEmbed[m.to])  — rank-POLICY_K bilinear
                    const K     = POLICY_K;
                    const fromL = policy.subarray(0, 64 * K);
                    const toL   = policy.subarray(64 * K, 128 * K);
                    let maxScore = -Infinity;
                    for (const m of entry.moves) {
                        let s = 0;
                        const fi = m.from * K, ti = m.to * K;
                        for (let k = 0; k < K; k++) s += fromL[fi + k] * toL[ti + k];
                        if (s > maxScore) maxScore = s;
                    }

                    entry.node.children = new Map();
                    let priorSum = 0;
                    const rawP = entry.moves.map(m => {
                        let s = 0;
                        const fi = m.from * K, ti = m.to * K;
                        for (let k = 0; k < K; k++) s += fromL[fi + k] * toL[ti + k];
                        const p = Math.exp(s - maxScore);
                        priorSum += p;
                        return p;
                    });
                    entry.moves.forEach((m, k) => {
                        entry.node.children.set(m.from * 64 + m.to, {
                            move: m,
                            node: new MCTSNode(rawP[k] / Math.max(priorSum, 1e-9)),
                        });
                    });
                }

                // Apply Dirichlet noise the first time the root gets expanded.
                if (!noiseApplied && nodeEvalIdx.has(root)) {
                    this._addRootNoise(root);
                    noiseApplied = true;
                }
            }

            // ── Backpropagation ───────────────────────────────────────────
            for (let s = 0; s < count; s++) {
                const { node, status } = allLeaves[s];
                let value;
                if (status !== 'playing') {
                    value = status === 'white_wins' ? 1 : status === 'black_wins' ? -1 : 0;
                } else {
                    const idx = nodeEvalIdx.get(node);
                    value = idx !== undefined ? toEval[idx].value : node.q;
                }
                for (const n of allPaths[s]) { n.visits++; n.valueSum += value; }
            }
        }

        return root;
    }

    // Add Dirichlet noise to root's child priors.
    // AlphaZero uses alpha=0.3, epsilon=0.25 at 800 sims.
    // We scale epsilon down proportionally so noise doesn't overwhelm few-sim searches.
    _addRootNoise(root, alpha = 0.3) {
        const epsilon = Math.min(0.25, 0.25 * (this.sims / 800) + 0.05);
        const children = [...root.children.values()];
        if (children.length === 0) return;
        const noise = _dirichletSample(alpha, children.length);
        children.forEach((entry, i) => {
            entry.node.prior = (1 - epsilon) * entry.node.prior + epsilon * noise[i];
        });
    }

    // UCB child selection: Q (from current player's perspective) + exploration bonus
    _selectChild(node, turn) {
        let best = -Infinity, bestEntry = null;
        const sqrtN = Math.sqrt(Math.max(node.visits, 1));

        for (const entry of node.children.values()) {
            const q = turn === WHITE ? entry.node.q : -entry.node.q;
            const u = this.cPuct * entry.node.prior * sqrtN / (1 + entry.node.visits);
            const score = q + u;
            if (score > best) { best = score; bestEntry = entry; }
        }
        return bestEntry;
    }

    // Pick a move from the root.
    // temperature=0:   deterministic (highest visit count)
    // temperature=1:   sample proportional to visit count
    // temperature=0.5: sample proportional to visits^2 (more peaked than temp=1)
    getBestMove(root, temperature = 0) {
        if (!root.children || root.children.size === 0) return null;
        const entries = [...root.children.values()];

        if (temperature === 0) {
            return entries.reduce((b, e) => e.node.visits > b.node.visits ? e : b).move;
        }
        // Sample proportional to visits^(1/temperature)
        const weights = entries.map(e => Math.pow(e.node.visits, 1 / temperature));
        const total = weights.reduce((s, w) => s + w, 0);
        let r = Math.random() * total, cum = 0;
        for (let i = 0; i < entries.length; i++) {
            cum += weights[i];
            if (r <= cum) return entries[i].move;
        }
        return entries[entries.length - 1].move;
    }

    // Return from-to visit-count distribution (4096-dim) for training.
    // Indexed by move.from * 64 + move.to.
    getPolicyTarget(root) {
        const pol = new Float32Array(POLICY_SIZE);
        if (!root.children) return pol;
        let total = 0;
        for (const { node } of root.children.values()) total += node.visits;
        for (const { move, node } of root.children.values())
            pol[move.from * 64 + move.to] += node.visits / total;
        return pol;
    }
}
