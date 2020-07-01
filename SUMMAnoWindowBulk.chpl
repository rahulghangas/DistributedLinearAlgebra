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

writeln(A.targetLocales());

const epsilon = 1e-8;
const refChecksum = (0.25*order*order*order*(order-1.0)*(order-1.0));

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

  // use VisualDebug;
  // startVdebug("comm");

  const AcolDim = A.domain.dim(1);
  const BrowDim = B.domain.dim(0);

  coforall loc in targetLocalesA {
    on loc {
      const localDomainA = A.localSubdomain();
      const localDomainB = B.localSubdomain();
      const localDomainC = C.localSubdomain();

      const AcolDim = A.domain.dim(1);
      const BrowDim = B.domain.dim(0);
      const subDomainA = {localDomainA.dim(0), AcolDim};
      const subDomainB = {BrowDim, localDomainB.dim(1)};
      const distsubDomainA : domain(2) dmapped Block(boundingBox = subDomainA, targetLocales=[loc]) = subDomainA;
      const distsubDomainB : domain(2) dmapped Block(boundingBox = subDomainB, targetLocales=[loc]) = subDomainB;
      var subArrayA : [distsubDomainA] A.eltType = A[localDomainA.dim(0), AcolDim];
      var subArrayB : [distsubDomainB] B.eltType = B[BrowDim, localDomainB.dim(1)];
      C[localDomainC] = dot(subArrayA, subArrayB);
    }
  }
  // stopVdebug();

  return C;
}

var t = new Timer();
t.start();
var result = distributedDot(A, B, windowSize);
t.stop();

writeln(t.elapsed());

const checksum = + reduce result;
if abs(checksum-refChecksum)/refChecksum > epsilon then
  halt("VALIDATION FAILED! Reference checksum = ", refChecksum,
                          " Checksum = ", checksum);
else
  writeln("Validation successful");
// writeln(result);
