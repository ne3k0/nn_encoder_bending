inlets = 7; // 0: control, 1: roll, 2: scale, 3: offset, 4: flip, 5: waveshape, 6: rollAxis
outlets = 2; // 0: bend_layer to nn~,  1: bang to visualizer

var rollAmt = 0;
var scaleAmt = 0;
var offsetAmt = 0;
var flipAmt = 0;
var shapeAmt = 0;
var rollAxis = 0;

var layerName = "";
var storedDims = [];
var original = null;

function layer_info(...args) {
    layerName  = args[0];
    var ndims  = args[2];
    storedDims = [];
    for (var i = 0; i < ndims; i++) storedDims.push(args[3 + i]);

    var buf = new Buffer("tensor");
    var n = buf.framecount();
    original = new Float32Array(n);
    for (var i = 0; i < n; i++) original[i] = buf.peek(1, i);
}

function msg_int(v) { msg_float(v); }
function msg_float(v) {
    if (inlet === 1) rollAmt   = v;
    else if (inlet === 2) scaleAmt  = v;
    else if (inlet === 3) offsetAmt = v;
    else if (inlet === 4) flipAmt = v;
    else if (inlet === 5) shapeAmt = v;
    else if (inlet === 6) rollAxis = Math.round(v);
    apply();
}

function bang() { apply(); }

function reset() {
    rollAmt = 0; scaleAmt = 0; offsetAmt = 0; flipAmt = 0; shapeAmt = 0;
    apply();
}

function apply() {
    if (!original || !layerName) return;

    var buf = new Buffer("tensor");
    var len = original.length;
    var bent = new Float32Array(len);

    rollBuffer(bent);

    for (var i = 0; i < len; i++) {
        var x = bent[i];
        x += offsetAmt;
        x *= scaleAmt;
        if (flipAmt) x = -x;
        if (shapeAmt > 0) x = x / (1.0 + shapeAmt * Math.abs(x));
        bent[i] = x;
    }

    for (var i = 0; i < len; i++) buf.poke(1, i, bent[i]);
    buf.send("dirty");

    outlet(0, "bend_layer", layerName, "tensor");
    outlet(1);
}

function bend_ok() { return; }

function rollBuffer(dst) {
    var len = original.length;

    if (rollAmt === 0) {
        for (var i = 0; i < len; i++) dst[i] = original[i];
        return;
    }

    if (storedDims.length === 1) {
        var n = Math.round(rollAmt * (len - 1)) % len;
        if (n < 0) n += len;
        for (var i = 0; i < len; i++) dst[i] = original[(i + n) % len];
        return;
    }

    var d0 = storedDims[0], d1 = storedDims[1], d2 = storedDims[2];
    var axLen = storedDims[rollAxis];
    var n = Math.round(rollAmt * (axLen - 1)) % axLen;
    if (n < 0) n += axLen;

    for (var i0 = 0; i0 < d0; i0++) {
        for (var i1 = 0; i1 < d1; i1++) {
            for (var i2 = 0; i2 < d2; i2++) {
                var dstIdx = i0 * (d1 * d2) + i1 * d2 + i2;
                var s0 = i0, s1 = i1, s2 = i2;
                if (rollAxis === 0) s0 = (i0 + n) % d0;
                else if (rollAxis === 1) s1 = (i1 + n) % d1;
                else s2 = (i2 + n) % d2;
                dst[dstIdx] = original[s0 * (d1 * d2) + s1 * d2 + s2];
            }
        }
    }
}
