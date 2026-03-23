'use strict';

const AUTO_PHASES = [
    { label: 'Random opponent',  mode: 'random', opSims: 1, thresh: 0.60, games: 30, matWeight: 0.30 },
    { label: 'Weak NN (1 sim)',  mode: 'weak',   opSims: 1, thresh: 0.60, games: 30, matWeight: 0.20 },
    { label: 'Weak NN (3 sims)', mode: 'weak',   opSims: 3, thresh: 0.55, games: 40, matWeight: 0.15 },
    { label: 'Weak NN (5 sims)', mode: 'weak',   opSims: 5, thresh: 0.55, games: 40, matWeight: 0.10 },
    { label: 'Self-play',        mode: 'self',   opSims: 1, thresh: null,  games: 60, matWeight: 0.05 },
];

class ChessUI {
    constructor(rootId) {
        this.root = document.getElementById(rootId);
        this.game = new Chess();
        this.nn = new NeuralNetwork();       // always start with a fresh model
        this.trainer = new Trainer(this.nn);
        this.mcts = new MCTS(this.nn, 50);  // for human-vs-AI play
        this.playerColor = WHITE;
        this.selectedSq = null;
        this.lossHistory = [];
        this._wakeLock = null;
        this._keepAwake = false;
        this._autoRunning = false;
        this._autoPhase = 0;
        this._buildDOM();
        this._renderBoard();
        // Try to restore previously saved weights and replay buffer in the background
        Promise.all([NeuralNetwork.load(), ReplayBuffer.load()]).then(([loaded, buffer]) => {
            if (loaded) {
                this.nn = loaded;
                this.trainer.nn = loaded;
                this.mcts.nn = loaded;
            }
            if (buffer) this.trainer.buffer = buffer;
            if (loaded || buffer) {
                const msg = [loaded && 'weights', buffer && `buffer (${buffer.size} examples)`]
                    .filter(Boolean).join(' + ');
                document.getElementById('status-bar').textContent = `Saved ${msg} loaded.`;
            }
        });
    }

    _buildDOM() {
        this.root.innerHTML = `
<div class="app">
  <div class="left-panel">
    <div class="board-outer">
      <div class="rank-labels">
        <span>8</span><span>7</span><span>6</span><span>5</span>
        <span>4</span><span>3</span><span>2</span><span>1</span>
      </div>
      <div class="board-col">
        <div id="board" class="board"></div>
        <div class="file-labels">
          <span>a</span><span>b</span><span>c</span><span>d</span>
          <span>e</span><span>f</span><span>g</span><span>h</span>
        </div>
      </div>
    </div>
    <div id="status-bar" class="status-bar">White to move</div>
  </div>

  <div class="right-panel">
    <div class="panel-header">
      <h1>Neural Chess</h1>
      <span class="subtitle">AlphaZero-style self-play</span>
    </div>

    <section>
      <h2>Game</h2>
      <div class="row">
        <button id="btn-new" class="btn-primary">New Game</button>
        <label>Play as
          <select id="sel-color">
            <option value="1">White</option>
            <option value="-1">Black</option>
          </select>
        </label>
        <label>AI sims
          <input id="sld-sims" type="range" min="10" max="200" value="50" step="10">
          <span id="lbl-sims">50</span>
        </label>
      </div>
    </section>

    <section>
      <h2>Training</h2>
      <div class="row">
        <label>Games <input id="inp-games" type="number" value="50" min="1" max="2000" style="width:58px"></label>
        <label>Sims/move <input id="inp-tsims" type="number" value="25" min="5" max="100" style="width:46px"></label>
        <label>Batch <input id="inp-batch" type="number" value="256" min="8" max="512" style="width:46px"></label>
      </div>
      <div class="row">
        <label>Opponent
          <select id="sel-opponent">
            <option value="self">Self-play</option>
            <option value="random">Random moves</option>
            <option value="weak">Weak NN</option>
          </select>
        </label>
        <label>Weak sims <input id="inp-oppsims" type="number" value="1" min="1" max="10" style="width:38px"></label>
      </div>
      <div class="sep"></div>
      <div class="row">
        <label>Decisive bias
          <input id="sld-bias" type="range" min="0" max="3" value="0" step="0.5">
          <span id="lbl-bias">0</span>
        </label>
        <label>Material wt
          <input id="sld-mat" type="range" min="0" max="0.6" value="0.1" step="0.1">
          <span id="lbl-mat">0.1</span>
        </label>
      </div>
      <div class="row">
        <label>Draw penalty
          <input id="sld-draw" type="range" min="0" max="0.3" value="0.05" step="0.05">
          <span id="lbl-draw">0.05</span>
        </label>
        <label>Opening plies <input id="inp-opening" type="number" value="0" min="0" max="20" style="width:40px"></label>
      </div>
      <div class="sep"></div>
      <div class="row">
        <button id="btn-train" class="btn-primary">▶ Train</button>
        <button id="btn-auto" class="btn-secondary">⟳ Auto-train</button>
        <button id="btn-stop" class="btn-secondary">■ Stop</button>
        <button id="btn-wakelock" class="btn-wakelock" title="Prevent the computer from sleeping during training">☽ Keep awake</button>
      </div>
      <div id="train-progress" style="display:none">
        <div class="progress-header">
          <span class="training-indicator"><span class="training-dot"></span><span id="lbl-phase">Training</span></span>
          <span id="lbl-progress">0 / 0</span>
        </div>
        <div class="progress-bar-wrap"><div id="progress-bar" class="progress-bar"></div></div>
      </div>
      <div class="row model-actions">
        <button id="btn-save" class="btn-secondary">Save</button>
        <button id="btn-load" class="btn-secondary">Load</button>
        <button id="btn-export" class="btn-secondary">Export</button>
        <button id="btn-import" class="btn-secondary">Import</button>
        <button id="btn-reset" class="btn-danger">Reset</button>
        <input id="inp-import" type="file" accept=".json,.bin" multiple style="display:none">
      </div>
    </section>

    <section>
      <h2>Stats</h2>
      <div class="stats-grid">
        <span>Games trained</span><span id="stat-games">0</span>
        <span>W / D / L (last 100)</span><span id="stat-wdl"><span class="win">0</span> / <span class="draw">0</span> / <span class="loss">0</span></span>
        <span>Game length (avg 20)</span><span id="stat-gamelength">—</span>
        <span>Buffer</span><span id="stat-buf">0</span>
        <span>Loss (avg 20)</span><span id="stat-loss">—</span>
        <span>Learn rate</span><span id="stat-lr">0.001</span>
        <span>Games/min</span><span id="stat-gpm">—</span>
      </div>
      <canvas id="outcome-canvas" width="280" height="30" style="display:block;margin-top:6px;border-radius:3px"></canvas>
      <canvas id="loss-canvas" width="280" height="100" style="margin-top:6px;border-radius:3px"></canvas>
    </section>
  </div>
</div>`;

        // Build board squares (rank 7 at top)
        const boardEl = document.getElementById('board');
        for (let r = 7; r >= 0; r--) {
            for (let f = 0; f < 8; f++) {
                const cell = document.createElement('div');
                const s = sq(r, f);
                cell.className = 'cell ' + ((r + f) % 2 === 0 ? 'dark' : 'light');
                cell.dataset.sq = s;
                cell.addEventListener('click', () => this._handleClick(s));
                boardEl.appendChild(cell);
            }
        }

        this._bindControls();
    }

    _bindControls() {
        document.getElementById('btn-new').addEventListener('click', () => this._newGame());
        document.getElementById('btn-train').addEventListener('click', () => this._startTraining());
        document.getElementById('btn-auto').addEventListener('click', () => {
            if (this._autoRunning) {
                this._autoRunning = false;
                this.trainer.stop();
            } else {
                this._autoTrain();
            }
        });
        document.getElementById('btn-stop').addEventListener('click', () => {
            this._autoRunning = false;
            this.trainer.stop();
        });
        document.getElementById('btn-save').addEventListener('click', async () => {
            try {
                await Promise.all([this.nn.save(), this.trainer.buffer.save()]);
                alert('Weights + replay buffer saved.');
            } catch (e) {
                alert('Save failed: ' + e.message);
            }
        });
        document.getElementById('btn-load').addEventListener('click', async () => {
            const loaded = await NeuralNetwork.load();
            if (loaded) {
                this.nn = loaded;
                this.trainer.nn = loaded;
                this.mcts.nn = loaded;
                alert('Weights loaded.');
            } else alert('No saved weights found.');
        });
        document.getElementById('btn-export').addEventListener('click', async () => {
            try {
                await this.nn.download();
            } catch (e) {
                alert('Export failed: ' + e.message);
            }
        });
        document.getElementById('btn-import').addEventListener('click', () => {
            document.getElementById('inp-import').value = '';
            document.getElementById('inp-import').click();
        });
        document.getElementById('btn-reset').addEventListener('click', async () => {
            if (!confirm('Delete all saved weights and replay buffer? This cannot be undone.')) return;
            await new Promise((resolve, reject) => {
                const req = indexedDB.deleteDatabase('chess_replay_v2');
                req.onsuccess = resolve; req.onerror = () => reject(req.error);
            });
            await tf.io.removeModel('indexeddb://chess_nn_v7').catch(() => {});
            const nn = new NeuralNetwork();
            this.nn = nn;
            this.trainer = new Trainer(nn);
            this.mcts.nn = nn;
            this.lossHistory = [];
            document.getElementById('stat-games').textContent  = '0';
            document.getElementById('stat-wdl').innerHTML      = '<span class="win">0</span> / <span class="draw">0</span> / <span class="loss">0</span>';
            document.getElementById('stat-gamelength').textContent = '—';
            document.getElementById('stat-buf').textContent    = '0';
            document.getElementById('stat-loss').textContent   = '—';
            document.getElementById('stat-lr').textContent     = '5.0e-4';
            document.getElementById('stat-gpm').textContent    = '—';
            document.getElementById('status-bar').textContent  = 'Model reset. Ready to train.';
            this._drawOutcomeChart();
            this._drawLossChart();
        });
        document.getElementById('inp-import').addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            const jsonFile    = files.find(f => f.name.endsWith('.json'));
            const weightsFile = files.find(f => f.name.endsWith('.bin'));
            if (!jsonFile || !weightsFile) {
                alert('Please select both the .json and .bin files exported by "Export model".');
                return;
            }
            try {
                const loaded = await NeuralNetwork.loadFromFiles(jsonFile, weightsFile);
                this.nn = loaded;
                this.trainer.nn = loaded;
                this.mcts.nn = loaded;
                alert('Model imported successfully.');
            } catch (e) {
                alert('Import failed: ' + e.message);
            }
        });
        document.getElementById('sel-color').addEventListener('change', e => {
            this.playerColor = parseInt(e.target.value);
        });
        document.getElementById('sld-sims').addEventListener('input', e => {
            document.getElementById('lbl-sims').textContent = e.target.value;
            this.mcts.sims = parseInt(e.target.value);
        });
        document.getElementById('sld-bias').addEventListener('input', e => {
            document.getElementById('lbl-bias').textContent = e.target.value;
            this.trainer.decisionBias = parseFloat(e.target.value);
        });
        document.getElementById('sld-mat').addEventListener('input', e => {
            document.getElementById('lbl-mat').textContent = e.target.value;
            this.trainer.materialWeight = parseFloat(e.target.value);
        });
        document.getElementById('sld-draw').addEventListener('input', e => {
            document.getElementById('lbl-draw').textContent = e.target.value;
            this.trainer.drawPenalty = parseFloat(e.target.value);
        });
        document.getElementById('sel-opponent').addEventListener('change', e => {
            this.trainer.opponentMode = e.target.value;
        });
        document.getElementById('inp-oppsims').addEventListener('change', e => {
            this.trainer.opponentSims = parseInt(e.target.value) || 1;
        });
        document.getElementById('btn-wakelock').addEventListener('click', async () => {
            this._keepAwake = !this._keepAwake;
            if (this._keepAwake) await this._requestWakeLock();
            else                  await this._releaseWakeLock();
        });
        document.addEventListener('visibilitychange', async () => {
            if (document.visibilityState === 'visible' && this._keepAwake && !this._wakeLock) {
                await this._requestWakeLock();
            }
        });
        document.getElementById('inp-opening').addEventListener('change', e => {
            this.trainer.openingPlies = parseInt(e.target.value) || 0;
        });
    }

    async _requestWakeLock() {
        if (!('wakeLock' in navigator)) return;
        try {
            this._wakeLock = await navigator.wakeLock.request('screen');
            this._wakeLock.addEventListener('release', () => {
                this._wakeLock = null;
                this._syncWakeLockBtn();
            });
        } catch (_) { /* denied or not supported */ }
        this._syncWakeLockBtn();
    }

    async _releaseWakeLock() {
        if (this._wakeLock) await this._wakeLock.release();
        // the 'release' event above clears this._wakeLock and calls _syncWakeLockBtn
    }

    _syncWakeLockBtn() {
        const btn = document.getElementById('btn-wakelock');
        if (!btn) return;
        const on = this._keepAwake && this._wakeLock !== null;
        btn.classList.toggle('wakelock-on', on);
        btn.textContent = on ? '☀ Keeping awake' : '☽ Keep awake';
    }

    _newGame() {
        this.game.reset();
        this.selectedSq = null;
        this._renderBoard();
        if (this.game.turn !== this.playerColor) {
            setTimeout(() => this._aiMove(), 150);
        }
    }

    _handleClick(square) {
        if (this.game.status() !== 'playing') return;
        if (this.game.turn !== this.playerColor) return;

        const piece = this.game.board[square];

        if (this.selectedSq === null) {
            if (piece && Math.sign(piece) === this.playerColor) {
                this.selectedSq = square;
                this._renderBoard();
            }
            return;
        }

        // Try to make the selected move
        const moves = this.game.legalMoves().filter(m => m.from === this.selectedSq && m.to === square);
        if (moves.length > 0) {
            // If promotion, prefer queen
            const m = moves.find(mv => mv.promo === this.playerColor * 5) || moves[0];
            this.game.makeMove(m);
            this.selectedSq = null;
            this._renderBoard();
            if (this.game.status() === 'playing') {
                setTimeout(() => this._aiMove(), 80);
            }
        } else if (piece && Math.sign(piece) === this.playerColor) {
            // Reselect
            this.selectedSq = square;
            this._renderBoard();
        } else {
            this.selectedSq = null;
            this._renderBoard();
        }
    }

    _aiMove() {
        if (this.game.status() !== 'playing') return;
        if (this.game.turn === this.playerColor) return;
        const root = this.mcts.search(this.game);
        const m = this.mcts.getBestMove(root, 0); // temperature=0 for deterministic play
        if (m) this.game.makeMove(m);
        this._renderBoard();
    }

    _renderBoard() {
        const legalTos = new Set(
            this.selectedSq !== null
                ? this.game.legalMoves().filter(m => m.from === this.selectedSq).map(m => m.to)
                : []
        );

        document.querySelectorAll('.cell').forEach(cell => {
            const s = parseInt(cell.dataset.sq);
            const p = this.game.board[s];
            cell.textContent = p ? (PIECE_SYMBOLS[String(p)] || '') : '';
            cell.classList.toggle('selected', s === this.selectedSq);
            cell.classList.toggle('legal', legalTos.has(s));
            cell.classList.toggle('piece-white', p > 0);
            cell.classList.toggle('piece-black', p < 0);
        });

        const s = this.game.status();
        const statusText = {
            playing:    `${this.game.turn === WHITE ? 'White' : 'Black'} to move`,
            white_wins: 'Checkmate — White wins!',
            black_wins: 'Checkmate — Black wins!',
            stalemate:  'Stalemate — Draw',
            draw_50:    '50-move rule — Draw',
            draw_rep:   'Threefold repetition — Draw',
            draw_material: 'Insufficient material — Draw',
        }[s] || s;
        document.getElementById('status-bar').textContent = statusText;
    }

    // Shared progress callback used by both manual and auto training.
    _onProgress(done, total, result) {
        const t = this.trainer;
        document.getElementById('lbl-progress').textContent = `${done} / ${total}`;
        document.getElementById('progress-bar').style.width = (done / total * 100) + '%';
        document.getElementById('stat-games').textContent = t.gamesPlayed;
        document.getElementById('stat-buf').textContent   = t.buffer.size;
        const avgLoss = t.recentLosses.length > 0
            ? t.recentLosses.reduce((s, v) => s + v, 0) / t.recentLosses.length
            : null;
        document.getElementById('stat-loss').textContent  =
            isNaN(result.loss)  ? 'train error (check console)'
          : avgLoss === null     ? 'filling…'
          :                       avgLoss.toFixed(5);
        const recent = t.recentOutcomes;
        const rW = recent.filter(o => o === 1).length;
        const rL = recent.filter(o => o === -1).length;
        const rD = recent.length - rW - rL;
        document.getElementById('stat-wdl').innerHTML =
            `<span class="win">${rW}</span> / <span class="draw">${rD}</span> / <span class="loss">${rL}</span>`;
        const avgLen = t.recentLengths.length > 0
            ? Math.round(t.recentLengths.reduce((s, v) => s + v, 0) / t.recentLengths.length)
            : 0;
        document.getElementById('stat-gamelength').textContent =
            avgLen > 0 ? `${avgLen} plies` : '—';
        document.getElementById('stat-lr').textContent = t._lr.toExponential(1);
        document.getElementById('stat-gpm').textContent =
            t.running ? `${t.gamesPerMin}/min` : '—';
        // Auto-save every 100 games so a crash loses at most 100 games of progress.
        if (t.gamesPlayed > 0 && t.gamesPlayed % 100 === 0) {
            Promise.all([this.nn.save(), this.trainer.buffer.save()])
                .catch(e => console.warn('Auto-save failed:', e));
        }

        if (isNaN(result.loss)) {
            this.trainer.stop();
            this._autoRunning = false;
            document.getElementById('status-bar').textContent = 'Training auto-stopped: NaN loss. Reset the model.';
        } else if (result.loss !== null) {
            this.lossHistory.push(result.loss);
            if (this.lossHistory.length > 200) this.lossHistory.shift();
            this._drawLossChart();

            // Auto-stop if loss oscillates too wildly. Use mean absolute difference
            // between consecutive losses — this catches oscillation but not smooth
            // declines (which would falsely trigger a std-based check).
            if (this.lossHistory.length >= 20) {
                const w = this.lossHistory.slice(-20);
                let diffSum = 0;
                for (let i = 1; i < w.length; i++) diffSum += Math.abs(w[i] - w[i - 1]);
                const meanDiff = diffSum / (w.length - 1);
                if (meanDiff > 0.4) {
                    this.trainer.stop();
                    this._autoRunning = false;
                    document.getElementById('status-bar').textContent =
                        `Training auto-stopped: loss too erratic (avg swing=${meanDiff.toFixed(3)}). Try resetting or lowering LR.`;
                }
            }
        }
        this._drawOutcomeChart();
    }

    _showProgress(label, numGames) {
        document.getElementById('lbl-phase').textContent = label;
        document.getElementById('lbl-progress').textContent = `0 / ${numGames}`;
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('train-progress').style.display = '';
    }

    async _startTraining() {
        if (this.trainer.running || this._autoRunning) return;
        const numGames = parseInt(document.getElementById('inp-games').value)  || 50;
        const sims     = parseInt(document.getElementById('inp-tsims').value)  || 25;
        const batch    = parseInt(document.getElementById('inp-batch').value)  || 256;

        if (this._keepAwake && !this._wakeLock) await this._requestWakeLock();
        this._showProgress('Training', numGames);

        await this.trainer.trainBatch(
            numGames,
            { sims, batchSize: batch },
            (done, total, result) => this._onProgress(done, total, result)
        );

        document.getElementById('train-progress').style.display = 'none';
    }

    async _autoTrain() {
        if (this.trainer.running) return;
        this._autoRunning = true;
        const btnAuto = document.getElementById('btn-auto');
        btnAuto.textContent = '⏹ Stop auto';
        btnAuto.classList.add('btn-auto-on');

        const sims  = parseInt(document.getElementById('inp-tsims').value) || 25;
        const batch = parseInt(document.getElementById('inp-batch').value) || 256;

        while (this._autoRunning) {
            const phase = AUTO_PHASES[this._autoPhase];

            // Apply phase settings to trainer
            this.trainer.opponentMode  = phase.mode;
            this.trainer.opponentSims  = phase.opSims;
            this.trainer.materialWeight = phase.matWeight;
            document.getElementById('sel-opponent').value = phase.mode;
            document.getElementById('inp-oppsims').value  = phase.opSims;
            document.getElementById('sld-mat').value      = phase.matWeight;
            document.getElementById('lbl-mat').textContent = phase.matWeight;

            if (this._keepAwake && !this._wakeLock) await this._requestWakeLock();

            const phaseLabel = `Phase ${this._autoPhase + 1}/${AUTO_PHASES.length}: ${phase.label}`;
            this._showProgress(phaseLabel, phase.games);

            await this.trainer.trainBatch(
                phase.games,
                { sims, batchSize: batch },
                (done, total, result) => this._onProgress(done, total, result)
            );

            if (!this._autoRunning) break;

            // Evaluate win rate over the games just played
            const recent   = this.trainer.recentOutcomes.slice(-phase.games);
            const winRate  = recent.filter(o => o === 1).length / Math.max(recent.length, 1);
            const drawRate = recent.filter(o => o === 0).length / Math.max(recent.length, 1);

            const isLastPhase = this._autoPhase === AUTO_PHASES.length - 1;
            const advance = isLastPhase || phase.thresh === null || winRate >= phase.thresh;
            const retreat = !advance && winRate + drawRate < 0.30 && this._autoPhase > 0;

            if (advance)       this._autoPhase = isLastPhase ? Math.max(0, AUTO_PHASES.length - 2) : this._autoPhase + 1;
            else if (retreat)  this._autoPhase--;
            // else stay on current phase
        }

        document.getElementById('train-progress').style.display = 'none';
        document.getElementById('lbl-phase').textContent = 'Training';
        btnAuto.textContent = '⟳ Auto-train';
        btnAuto.classList.remove('btn-auto-on');
        this._autoRunning = false;
    }

    _drawOutcomeChart() {
        const canvas = document.getElementById('outcome-canvas');
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const outcomes = this.trainer.recentOutcomes;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, w, h);
        if (!outcomes.length) return;
        const barW = w / 100; // always scale to 100 slots
        outcomes.forEach((o, i) => {
            ctx.fillStyle = o === 1 ? '#22c55e' : o === -1 ? '#ef4444' : '#6b7280';
            ctx.fillRect(i * barW, 0, Math.max(barW - 0.5, 1), h);
        });
        // Label
        ctx.fillStyle = '#9ca3af';
        ctx.font = '9px monospace';
        ctx.fillText('last 100 games  ■ win  ■ draw  ■ loss', 2, h - 3);
    }

    _drawLossChart() {
        const canvas = document.getElementById('loss-canvas');
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const losses = this.lossHistory;
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, w, h);

        if (losses.length < 2) return;

        const minL = Math.min(...losses);
        const maxL = Math.max(...losses);
        const range = maxL - minL || 1e-8;

        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        losses.forEach((l, i) => {
            const x = (i / (losses.length - 1)) * w;
            const y = h - ((l - minL) / range) * (h - 8) - 4;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Axis labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px monospace';
        ctx.fillText(maxL.toFixed(4), 2, 12);
        ctx.fillText(minL.toFixed(4), 2, h - 4);
    }
}
