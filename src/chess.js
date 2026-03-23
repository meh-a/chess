'use strict';

const WHITE = 1, BLACK = -1;

// Piece values: positive = white, negative = black
// 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King

const PIECE_SYMBOLS = {
    '1': '♙', '2': '♘', '3': '♗', '4': '♖', '5': '♕', '6': '♔',
    '-1': '♟', '-2': '♞', '-3': '♝', '-4': '♜', '-5': '♛', '-6': '♚'
};

function sq(rank, file) { return rank * 8 + file; }
function sqRank(s) { return s >> 3; }
function sqFile(s) { return s & 7; }
function onBoard(rank, file) { return rank >= 0 && rank < 8 && file >= 0 && file < 8; }

class Chess {
    constructor() { this.reset(); }

    reset() {
        this.board = new Int8Array(64);
        this.turn = WHITE;
        this.castling = { wk: true, wq: true, bk: true, bq: true };
        this.enPassant = -1; // target square index, or -1
        this.halfMoves = 0;
        this.fullMoves = 1;
        this.noFiftyMoveRule = false; // set true in training to allow endgame conversion
        this.posHistory = new Map();
        this._setupBoard();
        this.posHistory.set(this._posKey(), 1);
    }

    _posKey() {
        const c = this.castling;
        return `${this.board}|${this.turn}|${+c.wk}${+c.wq}${+c.bk}${+c.bq}|${this.enPassant}`;
    }

    _setupBoard() {
        const b = this.board;
        const back = [4, 2, 3, 5, 6, 3, 2, 4];
        for (let f = 0; f < 8; f++) {
            b[sq(0, f)] = back[f];   // white back rank
            b[sq(1, f)] = 1;         // white pawns
            b[sq(6, f)] = -1;        // black pawns
            b[sq(7, f)] = -back[f];  // black back rank
        }
    }

    // Generate pseudo-legal moves (may leave own king in check)
    _pseudoLegal() {
        const moves = [];
        const b = this.board;
        const t = this.turn;

        for (let s = 0; s < 64; s++) {
            const p = b[s];
            if (!p || Math.sign(p) !== t) continue;

            const r = sqRank(s), f = sqFile(s);
            const ap = Math.abs(p);

            if (ap === 1) { // Pawn
                const dr = t; // WHITE moves +rank, BLACK moves -rank
                const nr = r + dr;
                if (nr >= 0 && nr < 8) {
                    // Single push
                    if (!b[sq(nr, f)]) {
                        this._pawnMove(moves, s, sq(nr, f), nr);
                        // Double push from starting rank
                        const startRank = t === WHITE ? 1 : 6;
                        if (r === startRank && !b[sq(nr + dr, f)]) {
                            moves.push({ from: s, to: sq(nr + dr, f) });
                        }
                    }
                    // Diagonal captures
                    for (const df of [-1, 1]) {
                        const nf = f + df;
                        if (nf < 0 || nf > 7) continue;
                        const ts = sq(nr, nf);
                        if (b[ts] && Math.sign(b[ts]) === -t) {
                            this._pawnMove(moves, s, ts, nr);
                        } else if (ts === this.enPassant) {
                            moves.push({ from: s, to: ts, ep: true });
                        }
                    }
                }
            } else if (ap === 2) { // Knight
                for (const [dr, df] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
                    const nr = r + dr, nf = f + df;
                    if (!onBoard(nr, nf)) continue;
                    const target = b[sq(nr, nf)];
                    if (!target || Math.sign(target) === -t)
                        moves.push({ from: s, to: sq(nr, nf) });
                }
            } else if (ap === 6) { // King
                for (const [dr, df] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
                    const nr = r + dr, nf = f + df;
                    if (!onBoard(nr, nf)) continue;
                    const target = b[sq(nr, nf)];
                    if (!target || Math.sign(target) === -t)
                        moves.push({ from: s, to: sq(nr, nf) });
                }
                // Castling
                const kr = t === WHITE ? 0 : 7;
                if (s === sq(kr, 4)) {
                    if (t === WHITE) {
                        if (this.castling.wk && !b[sq(kr,5)] && !b[sq(kr,6)] && b[sq(kr,7)] === 4)
                            moves.push({ from: s, to: sq(kr, 6), castle: 'wk' });
                        if (this.castling.wq && !b[sq(kr,3)] && !b[sq(kr,2)] && !b[sq(kr,1)] && b[sq(kr,0)] === 4)
                            moves.push({ from: s, to: sq(kr, 2), castle: 'wq' });
                    } else {
                        if (this.castling.bk && !b[sq(kr,5)] && !b[sq(kr,6)] && b[sq(kr,7)] === -4)
                            moves.push({ from: s, to: sq(kr, 6), castle: 'bk' });
                        if (this.castling.bq && !b[sq(kr,3)] && !b[sq(kr,2)] && !b[sq(kr,1)] && b[sq(kr,0)] === -4)
                            moves.push({ from: s, to: sq(kr, 2), castle: 'bq' });
                    }
                }
            }

            // Sliding pieces
            if (ap === 3 || ap === 5) { // Bishop / Queen - diagonals
                for (const [dr, df] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
                    let nr = r + dr, nf = f + df;
                    while (onBoard(nr, nf)) {
                        const ts = sq(nr, nf), target = b[ts];
                        if (!target) { moves.push({ from: s, to: ts }); }
                        else { if (Math.sign(target) === -t) moves.push({ from: s, to: ts }); break; }
                        nr += dr; nf += df;
                    }
                }
            }
            if (ap === 4 || ap === 5) { // Rook / Queen - straights
                for (const [dr, df] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                    let nr = r + dr, nf = f + df;
                    while (onBoard(nr, nf)) {
                        const ts = sq(nr, nf), target = b[ts];
                        if (!target) { moves.push({ from: s, to: ts }); }
                        else { if (Math.sign(target) === -t) moves.push({ from: s, to: ts }); break; }
                        nr += dr; nf += df;
                    }
                }
            }
        }
        return moves;
    }

    _pawnMove(moves, from, to, destRank) {
        if (destRank === 0 || destRank === 7) {
            // Promotion: generate queen, rook, bishop, knight options
            for (const pt of [5, 4, 3, 2]) {
                moves.push({ from, to, promo: pt * this.turn });
            }
        } else {
            moves.push({ from, to });
        }
    }

    // Check if a square is attacked by the given color
    isAttacked(square, byColor) {
        const b = this.board;
        const r = sqRank(square), f = sqFile(square);

        // Pawn attacks (pawns of byColor attack "forward" from their perspective)
        const pawnDir = -byColor;
        for (const df of [-1, 1]) {
            const ar = r + pawnDir, af = f + df;
            if (onBoard(ar, af) && b[sq(ar, af)] === byColor) return true;
        }
        // Knight attacks
        for (const [dr, df] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
            const ar = r + dr, af = f + df;
            if (onBoard(ar, af) && b[sq(ar, af)] === byColor * 2) return true;
        }
        // Diagonal attacks (bishop / queen)
        for (const [dr, df] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
            let ar = r + dr, af = f + df;
            while (onBoard(ar, af)) {
                const p = b[sq(ar, af)];
                if (p) { if (p === byColor * 3 || p === byColor * 5) return true; break; }
                ar += dr; af += df;
            }
        }
        // Straight attacks (rook / queen)
        for (const [dr, df] of [[-1,0],[1,0],[0,-1],[0,1]]) {
            let ar = r + dr, af = f + df;
            while (onBoard(ar, af)) {
                const p = b[sq(ar, af)];
                if (p) { if (p === byColor * 4 || p === byColor * 5) return true; break; }
                ar += dr; af += df;
            }
        }
        // King attacks
        for (const [dr, df] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) {
            const ar = r + dr, af = f + df;
            if (onBoard(ar, af) && b[sq(ar, af)] === byColor * 6) return true;
        }
        return false;
    }

    findKing(color) {
        const king = color * 6;
        for (let s = 0; s < 64; s++) if (this.board[s] === king) return s;
        return -1;
    }

    inCheck(color) {
        const ks = this.findKing(color);
        return ks !== -1 && this.isAttacked(ks, -color);
    }

    legalMoves() {
        return this._pseudoLegal().filter(m => {
            // Castling: king cannot start in check or pass through an attacked square
            if (m.castle) {
                const kr = this.turn === WHITE ? 0 : 7;
                const passFile = m.castle.endsWith('k') ? 5 : 3;
                if (this.isAttacked(sq(kr, 4), -this.turn)) return false;
                if (this.isAttacked(sq(kr, passFile), -this.turn)) return false;
            }
            const undo = this.makeMove(m);
            const wasLegal = !this.inCheck(-this.turn); // check the side that just moved
            this.undoMove(undo);
            return wasLegal;
        });
    }

    makeMove(m) {
        const b = this.board;
        const undo = {
            m,
            captured: b[m.to],
            castling: { ...this.castling },
            enPassant: this.enPassant,
            halfMoves: this.halfMoves,
            turn: this.turn,
        };

        const ap = Math.abs(b[m.from]);

        // Update en passant target
        this.enPassant = -1;
        if (ap === 1 && Math.abs(m.to - m.from) === 16) {
            this.enPassant = (m.from + m.to) >> 1;
        }

        // En passant capture: remove the captured pawn
        if (m.ep) {
            const capSq = m.to - this.turn * 8;
            undo.epSq = capSq;
            undo.epPiece = b[capSq];
            b[capSq] = 0;
            undo.captured = 0;
        }

        // Castling: move the rook
        if (m.castle) {
            const rookPos = { wk:[sq(0,7),sq(0,5)], wq:[sq(0,0),sq(0,3)], bk:[sq(7,7),sq(7,5)], bq:[sq(7,0),sq(7,3)] }[m.castle];
            b[rookPos[1]] = b[rookPos[0]];
            b[rookPos[0]] = 0;
        }

        // Move the piece (handle promotion)
        b[m.to] = m.promo !== undefined ? m.promo : b[m.from];
        b[m.from] = 0;

        // Update castling rights
        if (ap === 6) {
            if (this.turn === WHITE) { this.castling.wk = false; this.castling.wq = false; }
            else                     { this.castling.bk = false; this.castling.bq = false; }
        }
        if (m.from === sq(0,0) || m.to === sq(0,0)) this.castling.wq = false;
        if (m.from === sq(0,7) || m.to === sq(0,7)) this.castling.wk = false;
        if (m.from === sq(7,0) || m.to === sq(7,0)) this.castling.bq = false;
        if (m.from === sq(7,7) || m.to === sq(7,7)) this.castling.bk = false;

        // Half-move clock (50-move rule)
        if (ap === 1 || undo.captured) this.halfMoves = 0;
        else this.halfMoves++;

        if (this.turn === BLACK) this.fullMoves++;
        this.turn = -this.turn;

        const k = this._posKey();
        this.posHistory.set(k, (this.posHistory.get(k) || 0) + 1);

        return undo;
    }

    undoMove(undo) {
        const b = this.board;
        const m = undo.m;

        // Decrement count for the position we're leaving
        const k = this._posKey();
        const cnt = (this.posHistory.get(k) || 1) - 1;
        if (cnt <= 0) this.posHistory.delete(k);
        else this.posHistory.set(k, cnt);

        this.turn = undo.turn;
        this.castling = undo.castling;
        this.enPassant = undo.enPassant;
        this.halfMoves = undo.halfMoves;
        if (this.turn === BLACK) this.fullMoves--;

        // Restore piece at origin (pawn if this was a promotion)
        b[m.from] = m.promo !== undefined ? this.turn : b[m.to];
        b[m.to] = undo.captured;

        if (m.ep) {
            b[undo.epSq] = undo.epPiece;
            b[m.to] = 0;
        }

        if (m.castle) {
            const rookPos = { wk:[sq(0,7),sq(0,5)], wq:[sq(0,0),sq(0,3)], bk:[sq(7,7),sq(7,5)], bq:[sq(7,0),sq(7,3)] }[m.castle];
            b[rookPos[0]] = b[rookPos[1]];
            b[rookPos[1]] = 0;
        }
    }

    status() {
        const moves = this.legalMoves();
        if (moves.length === 0) {
            return this.inCheck(this.turn)
                ? (this.turn === WHITE ? 'black_wins' : 'white_wins')
                : 'stalemate';
        }
        if (!this.noFiftyMoveRule && this.halfMoves >= 100) return 'draw_50';
        if ((this.posHistory.get(this._posKey()) || 0) >= 3) return 'draw_rep';
        // Bare kings (simplified insufficient material)
        let pieces = 0;
        for (let s = 0; s < 64; s++) if (this.board[s]) pieces++;
        if (pieces <= 2) return 'draw_material';
        return 'playing';
    }

    // Encode board state as a 910-float input vector for the NN.
    // [  0, 768): 12 piece-type planes × 64 squares (one-hot)
    // [768, 832): white attack plane (1 if white attacks that square)
    // [832, 896): black attack plane
    // [896, 900): castling rights (wk, wq, bk, bq)
    // [900, 908): en-passant file one-hot (8 bits, all zero if none)
    // [908]:      side to move (1 = white)
    // [909]:      50-move clock normalised to [0, 1]
    encodePosition() {
        const inp = new Float32Array(910);
        // Piece planes
        for (let s = 0; s < 64; s++) {
            const p = this.board[s];
            if (!p) continue;
            const plane = p > 0 ? p - 1 : 6 + (-p - 1);
            inp[plane * 64 + s] = 1;
        }
        // Attack planes
        for (let s = 0; s < 64; s++) {
            if (this.isAttacked(s, WHITE)) inp[768 + s] = 1;
            if (this.isAttacked(s, BLACK)) inp[832 + s] = 1;
        }
        // Castling
        inp[896] = this.castling.wk ? 1 : 0;
        inp[897] = this.castling.wq ? 1 : 0;
        inp[898] = this.castling.bk ? 1 : 0;
        inp[899] = this.castling.bq ? 1 : 0;
        // En passant file
        if (this.enPassant !== -1) inp[900 + sqFile(this.enPassant)] = 1;
        // Side to move + 50-move clock
        inp[908] = this.turn === WHITE ? 1 : 0;
        inp[909] = this.halfMoves / 100;
        return inp;
    }

    clone() {
        const g = new Chess();
        g.board = new Int8Array(this.board);
        g.turn = this.turn;
        g.castling = { ...this.castling };
        g.enPassant = this.enPassant;
        g.halfMoves = this.halfMoves;
        g.fullMoves = this.fullMoves;
        g.posHistory = new Map(this.posHistory);
        return g;
    }
}
