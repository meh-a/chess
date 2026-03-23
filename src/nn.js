'use strict';

// Architecture: [8,8,28] → Conv2D(64,3×3)+ReLU → ResBlock(64)×3
//   value head : Conv2D(1,1×1)+ReLU → flatten[64] → Dense(64)+ReLU → Dense(1)+tanh
//   policy head: two independent Conv2D(1,1×1) → flatten[64] each → concat[128]
//               score(from,to) = fromLogits[from] + toLogits[to]
//               training target: sparse 4096-dim vector indexed by from*64+to
//
// Input preprocessing (flat 910 → spatial tensor):
//   pos[0..895]  = 14 planes × 64 squares → reshape [14,8,8] → transpose [8,8,14]
//   pos[896..909]= 14 scalars → broadcast to [8,8,14]
//   concat along channel axis → [8,8,28]
//
// Runs on the GPU via TensorFlow.js (WebGL backend).
// Workers receive extracted weights as plain Float32Arrays for CPU inference.

const INPUT_SIZE    = 910;
const TRUNK_FILTERS = 64;    // conv filters per layer
const POLICY_K      = 8;     // rank of bilinear policy head: K from-embeddings × K to-embeddings per square
const MODEL_POL_OUT = 128 * POLICY_K; // network output: POLICY_K×64 from-logits + POLICY_K×64 to-logits
const POLICY_SIZE   = 4096;  // training target space: from*64 + to
const SPATIAL_CH    = 28;    // 14 spatial planes + 14 broadcast scalar channels

class NeuralNetwork {
    constructor() {
        this.model     = this._build();
        // epsilon=1e-5 (vs default 1e-7) prevents the Adam denominator from
        // amplifying near-zero gradient variance into enormous weight updates.
        this.optimizer = tf.train.adam(0.0005, 0.9, 0.999, 1e-5);
    }

    _build() {
        // Model input is already preprocessed to [8, 8, SPATIAL_CH]
        const inp = tf.input({ shape: [8, 8, SPATIAL_CH], name: 'board_input' });

        // Stem
        let x = tf.layers.conv2d({
            filters: TRUNK_FILTERS, kernelSize: 3, padding: 'same',
            activation: 'relu', name: 'stem_c'
        }).apply(inp);

        // Three residual blocks
        x = this._resBlock(x, 'rb1');
        x = this._resBlock(x, 'rb2');
        x = this._resBlock(x, 'rb3');

        // Value head: 1×1 conv → flatten[64] → Dense(64) → Dense(1, tanh)
        let v = tf.layers.conv2d({
            filters: 1, kernelSize: 1, padding: 'same',
            activation: 'relu', name: 'vh_c'
        }).apply(x);
        v = tf.layers.flatten({ name: 'vh_flat' }).apply(v);
        v = tf.layers.dense({ units: 64, activation: 'relu', name: 'vh1' }).apply(v);
        v = tf.layers.dense({ units: 1,  activation: 'tanh', name: 'vh2' }).apply(v);

        // Policy head: two independent 1×1 convs, each producing POLICY_K channels per square.
        // score(from, to) = dot(fromEmbed[from], toEmbed[to])  — rank-POLICY_K bilinear form.
        // Concatenated output [128*POLICY_K] is split and reshaped in training/inference code.
        let phFrom = tf.layers.conv2d({
            filters: POLICY_K, kernelSize: 1, padding: 'same', name: 'ph_from'
        }).apply(x);
        phFrom = tf.layers.flatten({ name: 'ph_from_flat' }).apply(phFrom); // [64*POLICY_K]
        let phTo = tf.layers.conv2d({
            filters: POLICY_K, kernelSize: 1, padding: 'same', name: 'ph_to'
        }).apply(x);
        phTo = tf.layers.flatten({ name: 'ph_to_flat' }).apply(phTo); // [64*POLICY_K]
        const pol = tf.layers.concatenate({ name: 'ph_cat' }).apply([phFrom, phTo]); // [128*POLICY_K]

        return tf.model({ inputs: inp, outputs: [v, pol], name: 'chess_nn' });
    }

    _resBlock(inp, name) {
        const z1  = tf.layers.conv2d({ filters: TRUNK_FILTERS, kernelSize: 3, padding: 'same', name: `${name}_c1` }).apply(inp);
        const a1  = tf.layers.activation({ activation: 'relu', name: `${name}_a1` }).apply(z1);
        const z2  = tf.layers.conv2d({ filters: TRUNK_FILTERS, kernelSize: 3, padding: 'same', name: `${name}_c2` }).apply(a1);
        const sum = tf.layers.add({ name: `${name}_add` }).apply([z2, inp]);
        return tf.layers.activation({ activation: 'relu', name: `${name}_out` }).apply(sum);
    }

    // ── Inference (main thread, WebGL) ───────────────────────────────────────

    forward(pos) { return this.predict(pos); }

    predict(pos) {
        return tf.tidy(() => {
            const raw = tf.tensor2d(pos, [1, INPUT_SIZE]);
            const inp = this._toSpatial(raw, 1);
            const [vt, pt] = this.model.predict(inp);
            return { v: vt.dataSync()[0], policy: new Float32Array(pt.dataSync()) };
        });
    }

    // Evaluate multiple positions in a single GPU call.
    predictBatch(positions) {
        if (!positions.length) return [];
        const n = positions.length;
        const posArr = new Float32Array(n * INPUT_SIZE);
        for (let i = 0; i < n; i++) posArr.set(positions[i], i * INPUT_SIZE);
        return tf.tidy(() => {
            const raw = tf.tensor2d(posArr, [n, INPUT_SIZE]);
            const inp = this._toSpatial(raw, n);
            const [vt, pt] = this.model.predict(inp);
            const vs = vt.dataSync();
            const ps = pt.dataSync();
            return Array.from({ length: n }, (_, i) => ({
                v: vs[i],
                policy: new Float32Array(ps.buffer, i * MODEL_POL_OUT * 4, MODEL_POL_OUT),
            }));
        });
    }

    // ── Training (main thread, WebGL) ────────────────────────────────────────

    train(samples, policyWeight = 1.0) {
        const n = samples.length;
        const posArr = new Float32Array(n * INPUT_SIZE);
        const vArr   = new Float32Array(n);
        const polArr = new Float32Array(n * POLICY_SIZE);

        samples.forEach(({ pos, vtarget, ptarget }, i) => {
            posArr.set(pos,     i * INPUT_SIZE);
            vArr[i] = vtarget;
            polArr.set(ptarget, i * POLICY_SIZE);
        });

        const raw       = tf.tensor2d(posArr, [n, INPUT_SIZE]);
        const inpTensor = this._toSpatial(raw, n);
        raw.dispose();
        const vTensor   = tf.tensor2d(vArr,  [n, 1]);
        const polTensor = tf.tensor2d(polArr, [n, POLICY_SIZE]);

        let costTensor;
        try {
            costTensor = this.optimizer.minimize(() => {
                const preds  = this.model.apply(inpTensor, { training: true });
                const vLoss  = tf.losses.meanSquaredError(vTensor, preds[0]);
                // Reconstruct full 4096-dim logits via bilinear from+to heads.
                // logit[from, to] = dot(fromEmbed[from], toEmbed[to])
                //   fromL: [n, 64*K] → reshape [n, 64, K]
                //   toL:   [n, 64*K] → reshape [n, K, 64]
                //   matMul → [n, 64, 64] → reshape [n, 4096]
                // Each from-embedding contributes to 64 dot products (one per to-square),
                // so gradient is 64× amplified — divide by 64 to normalise scale.
                const K     = POLICY_K;
                const fromL = preds[1].slice([0,      0], [-1, 64 * K]).reshape([-1, 64, K]);
                const toL   = preds[1].slice([0, 64 * K], [-1, 64 * K]).reshape([-1, K, 64]);
                const polLogits = tf.matMul(fromL, toL).div(64).reshape([-1, POLICY_SIZE]); // [n, 4096]
                // Mask logits for illegal/unvisited moves (ptarget == 0) to -1e9 so the
                // softmax denominator only counts MCTS-visited legal moves.
                const penalty   = polTensor.greater(0).logicalNot().toFloat().mul(1e9);
                const polLoss   = tf.losses.softmaxCrossEntropy(polTensor, polLogits.sub(penalty)).mul(policyWeight);
                return vLoss.add(polLoss);
            }, true);
            return costTensor.dataSync()[0];
        } finally {
            // Always dispose GPU tensors — even if minimize() throws — to prevent
            // memory accumulation that would crash the tab during overnight runs.
            inpTensor.dispose();
            vTensor.dispose();
            polTensor.dispose();
            if (costTensor) costTensor.dispose();
        }
    }

    // Preprocess flat [n, 910] tensor → [n, 8, 8, 28]
    // Wrapped in tf.tidy() so the 6 intermediate tensors are disposed automatically.
    // The returned concat tensor escapes tidy and must be disposed by the caller.
    _toSpatial(raw, n) {
        return tf.tidy(() => {
            // 14 spatial planes × 64 squares → [n, 14, 8, 8] → [n, 8, 8, 14] (NHWC)
            const spatial = raw.slice([0, 0], [-1, 896])
                .reshape([n, 14, 8, 8])
                .transpose([0, 2, 3, 1]);
            // 14 scalars → broadcast to [n, 8, 8, 14]
            const scalars = raw.slice([0, 896], [-1, 14])
                .reshape([n, 1, 1, 14])
                .tile([1, 8, 8, 1]);
            return tf.concat([spatial, scalars], 3); // [n, 8, 8, 28]
        });
    }

    // ── Weight extraction (for Web Workers, CPU inference) ───────────────────
    // Conv2D kernels in TF.js: [kH, kW, Cin, Cout]
    // Worker format: [Cout, kH, kW, Cin] (row-major, o-major)

    extractWeights() {
        const xposeConv = (flat, kH, kW, Cin, Cout) => {
            const w = new Float32Array(Cout * kH * kW * Cin);
            for (let o = 0; o < Cout; o++)
                for (let dr = 0; dr < kH; dr++)
                    for (let dc = 0; dc < kW; dc++)
                        for (let ci = 0; ci < Cin; ci++)
                            w[((o * kH + dr) * kW + dc) * Cin + ci] =
                            flat[((dr * kW + dc) * Cin + ci) * Cout + o];
            return w;
        };

        // 1×1 conv and dense both use [Cout, Cin] layout
        const xposeDense = (flat, Cin, Cout) => {
            const w = new Float32Array(Cout * Cin);
            for (let o = 0; o < Cout; o++)
                for (let ci = 0; ci < Cin; ci++)
                    w[o * Cin + ci] = flat[ci * Cout + o];
            return w;
        };

        const getConv = (name, kH, kW, Cin, Cout) => {
            const [k, b] = this.model.getLayer(name).getWeights();
            return { w: xposeConv(k.dataSync(), kH, kW, Cin, Cout), b: new Float32Array(b.dataSync()) };
        };
        const get1x1 = (name, Cin, Cout) => {
            const [k, b] = this.model.getLayer(name).getWeights();
            return { w: xposeDense(k.dataSync(), Cin, Cout), b: new Float32Array(b.dataSync()) };
        };
        const getDense = (name, Cin, Cout) => {
            const [k, b] = this.model.getLayer(name).getWeights();
            return { w: xposeDense(k.dataSync(), Cin, Cout), b: new Float32Array(b.dataSync()) };
        };

        const T = TRUNK_FILTERS;
        return {
            stem_c: getConv('stem_c', 3, 3, SPATIAL_CH, T),
            rb1_c1: getConv('rb1_c1', 3, 3, T, T),
            rb1_c2: getConv('rb1_c2', 3, 3, T, T),
            rb2_c1: getConv('rb2_c1', 3, 3, T, T),
            rb2_c2: getConv('rb2_c2', 3, 3, T, T),
            rb3_c1: getConv('rb3_c1', 3, 3, T, T),
            rb3_c2: getConv('rb3_c2', 3, 3, T, T),
            vh_c:    get1x1('vh_c',    T, 1),  // [T→1] per square → flatten [64]
            vh1:     getDense('vh1',  64, 64),
            vh2:     getDense('vh2',  64, 1),
            ph_from: get1x1('ph_from', T, POLICY_K),  // [T→POLICY_K] per square → flatten [64*POLICY_K]
            ph_to:   get1x1('ph_to',   T, POLICY_K),  // [T→POLICY_K] per square → flatten [64*POLICY_K]
        };
    }

    setLearningRate(lr) {
        this.optimizer.learningRate = lr;
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    async save() {
        await this.model.save('indexeddb://chess_nn_v8');
    }

    async download() {
        await this.model.save('downloads://chess_nn');
    }

    static async loadFromFiles(jsonFile, weightsFile) {
        try {
            const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            const nn = new NeuralNetwork();
            nn.model = model;
            return nn;
        } catch (e) {
            throw new Error('Failed to load model files: ' + e.message);
        }
    }

    static async load() {
        try {
            const model = await tf.loadLayersModel('indexeddb://chess_nn_v8');
            const nn    = new NeuralNetwork();
            nn.model    = model;
            return nn;
        } catch (_) {
            return null;
        }
    }
}
