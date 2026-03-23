'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// COLOR-FLIP AUGMENTATION
// Mirror the position to the opponent's perspective so each game yields 2×
// training examples (original + flipped with negated value target).
// ─────────────────────────────────────────────────────────────────────────────

// Material score from White's perspective, normalised to [-1, 1].
// Max one-side material: 8×1 + 2×3 + 2×3 + 2×5 + 1×9 = 39
const _MAT_VALUES = [0, 1, 3, 3, 5, 9, 0]; // indexed by |piece|
function materialScore(board) {
    let score = 0;
    for (let i = 0; i < 64; i++) {
        const p = board[i];
        if (p) score += Math.sign(p) * _MAT_VALUES[Math.abs(p)];
    }
    return score / 39;
}

function flipPos(pos) {
    const f = new Float32Array(910);
    // Piece planes 0-5 (white) ↔ 6-11 (black), with rank mirror
    for (let plane = 0; plane < 12; plane++) {
        const src = plane < 6 ? plane + 6 : plane - 6;
        for (let s = 0; s < 64; s++) {
            const fs = (7 - (s >> 3)) * 8 + (s & 7);
            f[plane * 64 + s] = pos[src * 64 + fs];
        }
    }
    // Attack planes: white (768-831) ↔ black (832-895), rank mirror
    for (let s = 0; s < 64; s++) {
        const fs = (7 - (s >> 3)) * 8 + (s & 7);
        f[768 + s] = pos[832 + fs];
        f[832 + s] = pos[768 + fs];
    }
    // Castling: wk,wq ↔ bk,bq
    f[896] = pos[898]; f[897] = pos[899];
    f[898] = pos[896]; f[899] = pos[897];
    // En-passant file is symmetric
    for (let i = 0; i < 8; i++) f[900 + i] = pos[900 + i];
    // Side-to-move flips; half-move clock unchanged
    f[908] = 1 - pos[908];
    f[909] = pos[909];
    return f;
}

function flipPol(pol) {
    // pol is 4096-dim indexed by from*64+to.
    // Flipping ranks: square s → (7 - rank)*8 + file.
    // A move (from, to) on the flipped board becomes (flip(from), flip(to)).
    const f = new Float32Array(POLICY_SIZE);
    for (let from = 0; from < 64; from++) {
        const fFrom = (7 - (from >> 3)) * 8 + (from & 7);
        for (let to = 0; to < 64; to++) {
            const fTo = (7 - (to >> 3)) * 8 + (to & 7);
            f[fFrom * 64 + fTo] = pol[from * 64 + to];
        }
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// REPLAY BUFFER  –  circular, evicts oldest entries when full
// ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer {
    constructor(capacity = 20000) {
        this.capacity = capacity;
        this.buf  = [];
        this.head = 0;
    }
    push(pos, vtarget, ptarget) {
        const e = { pos, vtarget, ptarget };
        if (this.buf.length < this.capacity) { this.buf.push(e); }
        else { this.buf[this.head] = e; this.head = (this.head + 1) % this.capacity; }
    }
    sample(n) {
        const out = [], len = this.buf.length;
        for (let i = 0; i < n; i++) out.push(this.buf[Math.floor(Math.random() * len)]);
        return out;
    }

    // Sample weighted by |vtarget|^bias so decisive positions are preferred.
    // bias=0 falls back to uniform; bias=1 linear weight; bias=2 quadratic.
    sampleWeighted(n, bias = 0) {
        if (bias === 0) return this.sample(n);
        const len = this.buf.length;
        // Build CDF weighted by (0.1 + |vtarget|)^bias
        const cdf = new Float32Array(len);
        let total = 0;
        for (let i = 0; i < len; i++) {
            total += Math.pow(0.1 + Math.abs(this.buf[i].vtarget), bias);
            cdf[i] = total;
        }
        const out = [];
        for (let i = 0; i < n; i++) {
            const r = Math.random() * total;
            let lo = 0, hi = len - 1;
            while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
            out.push(this.buf[lo]);
        }
        return out;
    }
    get size() { return this.buf.length; }

    // Each entry is packed as 5007 floats: 910 (pos) + 1 (vtarget) + 4096 (ptarget)
    static get _ENTRY_SIZE() { return 5007; }

    async save() {
        const n = this.buf.length;
        const E = ReplayBuffer._ENTRY_SIZE;
        const packed = new Float32Array(n * E);
        for (let i = 0; i < n; i++) {
            const { pos, vtarget, ptarget } = this.buf[i];
            packed.set(pos, i * E);
            packed[i * E + 910] = vtarget;
            packed.set(ptarget, i * E + 911);
        }
        return new Promise((resolve, reject) => {
            const req = indexedDB.open('chess_replay_v2', 1);
            req.onupgradeneeded = e => e.target.result.createObjectStore('data');
            req.onsuccess = e => {
                const db = e.target.result;
                const tx = db.transaction('data', 'readwrite');
                tx.objectStore('data').put({ packed, head: this.head }, 'buf');
                tx.oncomplete = () => { db.close(); resolve(); };
                tx.onerror   = () => { db.close(); reject(tx.error); };
            };
            req.onerror = () => reject(req.error);
        });
    }

    static async load(capacity = 20000) {
        return new Promise(resolve => {
            const req = indexedDB.open('chess_replay_v2', 1);
            req.onupgradeneeded = e => e.target.result.createObjectStore('data');
            req.onsuccess = e => {
                const db = e.target.result;
                const tx = db.transaction('data', 'readonly');
                const get = tx.objectStore('data').get('buf');
                get.onsuccess = () => {
                    db.close();
                    const result = get.result;
                    if (!result) { resolve(null); return; }
                    const { packed, head } = result;
                    const E = ReplayBuffer._ENTRY_SIZE;
                    const n = packed.length / E;
                    const rb = new ReplayBuffer(capacity);
                    rb.head = head;
                    for (let i = 0; i < n; i++) {
                        rb.buf.push({
                            pos:     packed.slice(i * E,         i * E + 910),
                            vtarget: packed[i * E + 910],
                            ptarget: packed.slice(i * E + 911,   i * E + E),
                        });
                    }
                    resolve(rb);
                };
                get.onerror = () => { db.close(); resolve(null); };
            };
            req.onerror = () => resolve(null);
        });
    }
}

// Hash a position for repetition detection. Covers pieces, castling, en passant,
// and side-to-move — the same fields chess uses for threefold repetition.
function _hashPos(pos) {
    let h = 0;
    for (let i = 0; i < 768; i++) if (pos[i]) h = (h ^ (i * 2654435761)) >>> 0;
    for (let i = 896; i < 909; i++) if (pos[i]) h = (h ^ (i * 1000003)) >>> 0;
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAINER  –  runs self-play on the main thread using TF.js/WebGL
// ─────────────────────────────────────────────────────────────────────────────

class Trainer {
    constructor(nn) {
        this.nn          = nn;
        this.buffer      = new ReplayBuffer(20000);
        this.gamesPlayed = 0;
        this.running     = false;
        this.numWorkers  = 1;
        this.lastLoss    = null;
        this._batchSize  = 256;
        this._stepsPerGame = 2;
        this._lr          = 0.0005;
        this.decisionBias = 0;
        this.materialWeight    = 0.1;
        this.drawPenalty       = 0.05;
        this.repetitionPenalty = 0.15; // extra vtarget penalty for revisiting a position
        this.opponentMode    = 'self'; // 'self' | 'random' | 'weak'
        this.opponentSims    = 1;
        this.openingPlies    = 0;
        this.wins            = 0;
        this.losses          = 0;
        this.draws           = 0;
        this.lastGameLength  = 0;
        this.recentOutcomes  = []; // last 100 outcomes: 1, -1, or 0
        this.recentLengths   = []; // last 20 game lengths for rolling average
        this.recentLosses    = []; // last 20 loss values for rolling average
        this._trainStartTime  = 0;
        this._trainStartGames = 0;
    }

    async _playGame(sims) {
        const opMode = this.opponentMode; // 'self' | 'random' | 'weak'

        // In opponent mode, alternate which colour gets the strong MCTS each game
        // so the network learns from both sides equally.
        const strongSide = opMode !== 'self'
            ? (this.gamesPlayed % 2 === 0 ? WHITE : BLACK)
            : null; // null = self-play, both sides use full MCTS

        const mcts = new MCTS(this.nn, sims, 1.5, this._mctsMatWeight);
        const weakMcts = opMode === 'weak'
            ? new MCTS(this.nn, Math.max(1, this.opponentSims))
            : null;

        const game = new Chess();
        game.noFiftyMoveRule = true; // allow endgame conversion: 50-move draws prevent the NN
                                     // from ever seeing a win, starving the value head of signal
        const history = []; // only strong-side positions
        const positionCounts = new Map(); // hash → visit count for repetition detection
        let badStreakWhite = 0, badStreakBlack = 0;
        let resigned = false, resignOutcome = 0;
        let totalPlies = 0;

        // Random/weak opponents rarely deliver mate quickly; give them more room.
        const plyLimit = opMode !== 'self' ? 600 : 200;
        while (game.status() === 'playing' && totalPlies < plyLimit) {
            const pos = game.encodePosition();
            const mat = materialScore(game.board);
            const isStrongTurn = strongSide === null || game.turn === strongSide;
            const posHash = _hashPos(pos);
            const visitCount = (positionCounts.get(posHash) || 0) + 1;
            positionCounts.set(posHash, visitCount);
            const isRepeat = isStrongTurn && visitCount > 1;
            let move, rootQ = 0, ptarget = null;

            if (totalPlies < this.openingPlies) {
                // Random opening plies for both sides
                const legal = game.legalMoves();
                if (!legal.length) break;
                move = legal[Math.floor(Math.random() * legal.length)];
                if (isStrongTurn) ptarget = new Float32Array(POLICY_SIZE);

            } else if (!isStrongTurn) {
                // Opponent's turn — no training example recorded
                if (opMode === 'random') {
                    const legal = game.legalMoves();
                    if (!legal.length) break;
                    move = legal[Math.floor(Math.random() * legal.length)];
                } else { // 'weak'
                    const root = weakMcts.search(game, false);
                    move = weakMcts.getBestMove(root, 0.5);
                    if (!move) break;
                }
                // Advance the strong MCTS tree past the opponent's move so tree
                // reuse stays valid on the strong player's next turn.
                mcts.advanceRoot(move);

            } else {
                // Strong player's MCTS turn
                const root = mcts.search(game, true);
                rootQ = root.visits > 0 ? root.valueSum / root.visits : 0;
                const temp = totalPlies < 30 ? 1 : Math.max(0, 1 - (totalPlies - 30) / 60);
                move = mcts.getBestMove(root, temp);
                if (!move) break;
                ptarget = mcts.getPolicyTarget(root);

                if (opMode === 'self') {
                    // Self-play: resign only when clearly and consistently lost.
                    // Threshold -0.95 + streak 10 prevents the same death spiral
                    // that killed opponent mode: a miscalibrated value head triggers
                    // early resignations → more loss examples → more pessimism.
                    if (rootQ < -0.95) { badStreakWhite++; badStreakBlack = 0; }
                    else if (rootQ > 0.95) { badStreakBlack++; badStreakWhite = 0; }
                    else { badStreakWhite = 0; badStreakBlack = 0; }
                    if (badStreakWhite >= 10 && history.length > 40) {
                        resigned = true; resignOutcome = -1; break;
                    }
                    if (badStreakBlack >= 10 && history.length > 40) {
                        resigned = true; resignOutcome = 1; break;
                    }
                }
                // Opponent mode: never resign — always play to actual checkmate/draw.
                // Resigning would create a death spiral: a pessimistic value head
                // causes early resignations, which reinforce pessimism.

                mcts.advanceRoot(move);
            }

            if (isStrongTurn && ptarget !== null) {
                history.push({ pos, ptarget, rootQ, mat, isRepeat });
            }

            game.makeMove(move);
            totalPlies++;
            if (totalPlies % 10 === 0) await new Promise(r => setTimeout(r, 0));
        }

        const s = game.status();
        const lastEntry = history.length > 0 ? history[history.length - 1] : null;
        const finalQ    = lastEntry ? lastEntry.rootQ : 0;
        const lastMat   = lastEntry ? lastEntry.mat   : 0;
        // When the ply limit is hit, blend material (honest even when NN is untrained)
        // with rootQ (more accurate once training progresses) weighted by how far along
        // training is. Early: mostly material. Later: mostly rootQ.
        const mctsT = Math.min(1, this.gamesPlayed / 500);
        const plyLimitOutcome = lastMat * (1 - mctsT) + finalQ * mctsT;
        const outcome = resigned ? resignOutcome
            : s === 'white_wins' ? 1 : s === 'black_wins' ? -1
            : s === 'draw_rep'   ? -1  // treat as a loss — NN had material advantage and failed to convert
            : s !== 'playing'    ? -this.drawPenalty
            : plyLimitOutcome;

        const mw = this.materialWeight;
        const rp = this.repetitionPenalty;
        // vtarget blends outcome (ground truth), rootQ (per-position smoothing), and material.
        // rootQ weight (0.2) is kept small so outcome dominates and bias can't amplify
        // into a draw-attractor the way the old 0.45/0.45 split could.
        const ow = (1 - mw) * 0.75; // outcome weight  ≈ 0.675
        const qw = (1 - mw) * 0.25; // rootQ weight    ≈ 0.225
        const examples = history.map(({ pos, ptarget, rootQ, mat, isRepeat }) => ({
            pos,
            ptarget,
            vtarget: Math.min(1, Math.max(-1, ow * outcome + qw * rootQ + mw * mat - (isRepeat ? rp : 0))),
        }));
        // winner: the side that actually won by checkmate/resignation (0 = no decisive result)
        const winner = resigned ? resignOutcome : s === 'white_wins' ? 1 : s === 'black_wins' ? -1 : 0;
        return { outcome, examples, winner, strongSide };
    }

    async trainBatch(numGames, options = {}, progressCb = null) {
        if (this.running) return;
        this.running = true;
        this._batchSize    = options.batchSize    || 256;
        this._stepsPerGame = options.stepsPerGame || 4;
        const sims = options.sims || 25;
        this._trainStartTime  = Date.now();
        this._trainStartGames = this.gamesPlayed;

        for (let i = 0; i < numGames && this.running; i++) {
            const { outcome, examples, winner, strongSide } = await this._playGame(sims);

            for (const { pos, vtarget, ptarget } of examples) {
                this.buffer.push(pos, vtarget, ptarget);
                this.buffer.push(flipPos(pos), -vtarget, flipPol(ptarget));
            }

            if (this.buffer.size >= this._batchSize) {
                try {
                    for (let step = 0; step < this._stepsPerGame; step++) {
                        this.lastLoss = this.nn.train(this.buffer.sampleWeighted(this._batchSize, this.decisionBias), 0.5);
                    }
                } catch (e) {
                    console.error('Training step threw:', e);
                    this.lastLoss = NaN; // surface the failure in the UI
                }
            }

            this.gamesPlayed++;
            this.lastGameLength = examples.length;
            this.recentLengths.push(examples.length);
            if (this.recentLengths.length > 20) this.recentLengths.shift();
            if (this.lastLoss !== null && !isNaN(this.lastLoss)) {
                this.recentLosses.push(this.lastLoss);
                if (this.recentLosses.length > 20) this.recentLosses.shift();
            }
            // In opponent mode track W/D/L from the NN's perspective, not White's.
            const nnResult = strongSide !== null && winner !== 0
                ? (winner === strongSide ? 1 : -1)
                : winner;
            if (nnResult === 1) this.wins++;
            else if (nnResult === -1) this.losses++;
            else this.draws++;
            this.recentOutcomes.push(nnResult);
            if (this.recentOutcomes.length > 100) this.recentOutcomes.shift();

            // Learning rate decay: halve every 2000 games, floor at 1e-5
            const prevLrStep = Math.floor((this.gamesPlayed - 1) / 2000);
            const currLrStep = Math.floor(this.gamesPlayed / 2000);
            if (currLrStep > prevLrStep) {
                this._lr = Math.max(this._lr * 0.5, 1e-5);
                this.nn.setLearningRate(this._lr);
            }

            if (progressCb) progressCb(i + 1, numGames, { outcome, loss: this.lastLoss });
        }

        this.running = false;
    }

    // MCTS material weight starts at 0.3 to give early search meaningful signal
    // when the value head is untrained (output ≈ 0 everywhere), then decays
    // linearly to materialWeight over 500 games as the NN takes over.
    get _mctsMatWeight() {
        const start = Math.max(this.materialWeight, 0.30);
        const t     = Math.min(1, this.gamesPlayed / 500);
        return start + t * (this.materialWeight - start);
    }

    get gamesPerMin() {
        const elapsed = (Date.now() - this._trainStartTime) / 60000;
        if (elapsed < 0.05) return 0;
        return ((this.gamesPlayed - this._trainStartGames) / elapsed).toFixed(1);
    }

    stop() {
        this.running = false;
    }
}
