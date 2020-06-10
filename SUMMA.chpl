use Time;
use BlockDist;
// use RangeChunk;
use LinearAlgebra;

config type dtype = real;
config const order = 10;
config const windowSize = 10;

const vecRange = 0..#order;
const matrixSpace = {vecRange, vecRange};

const matrixDom : domain(2) dmapped Block(boundingBox = matrixSpace) = matrixSpace;

var A = Matrix(matrixDom, dtype),
    B = Matrix(matrixDom, dtype);

forall (i,j) in matrixDom {
  A[i,j] = j;
  B[i,j] = j;
}

private inline iter block(indexes, window : int) {
  var lowVal = indexes.low;
  while (true) {
    var highVal = lowVal + indexes.stride * window - 1;
    if (highVal >= indexes.high) {
      yield lowVal..indexes.high;
      break;
    } else {
      yield lowVal..highVal;
    }
    lowVal = highVal + 1;
  }
}

proc distributedDot(A : [] ?t, B : [] t, window : int = 500) {
  ref targetLocalesA = A.targetLocales();
  ref targetLocalesB = B.targetLocales();

  if (A.shape(1) != B.shape(0)) {
    halt("Array dimensions don't match.\n Trying to multiply arrays of dimensions ", 
         A.shape, " ", B.shape);
  }
  if (targetLocalesA.size != targetLocalesB.size) {
    halt("Matrices should be distributed on the same locales");
  }

  const targetLocales = reshape(targetLocalesA, 
                                {0..#targetLocalesA.shape(0), 
                                 0..#targetLocalesB.shape(1)});

  var domainC : domain(2) dmapped Block(boundingBox = {A.domain.dim(0), B.domain.dim(1)},
                                        targetLocales = targetLocales) 
                                        = {A.domain.dim(0), B.domain.dim(1)};
  var C : [domainC] A.eltType;

  coforall loc in targetLocalesA {
    on loc {
      var localDomainA = A.localSubdomain();
      var localDomainB = B.localSubdomain();
      var localDomainC = C.localSubdomain();

      const commonDimension = A.domain.dim(1);
      for subArrayAChunk in block(A.domain.dim(1), window) {
        // var subArrayAChunk = commonDimension # windowSize align i;
        var subArrayA : [localDomainA.dim(0), subArrayAChunk] A.eltType = A[localDomainA.dim(0), subArrayAChunk];
        var subArrayB : [subArrayAChunk, localDomainB.dim(1)] B.eltType = B[subArrayAChunk ,localDomainB.dim(1)];
        C[localDomainC] += dot(subArrayA, subArrayB);
      }
    }
  }

  return C;
}

var t = new Timer();
t.start();
var result = distributedDot(A, B, windowSize);
t.stop();

writeln(t.elapsed());
// writeln(result);
