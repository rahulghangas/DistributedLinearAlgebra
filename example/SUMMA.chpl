use Time;
use BlockDist;
use LinearAlgebra;


config const order = 10,

const vecRange = 0..#order;

const matrixSpace = {vecRange, vecRange};

config var N_1 = 2;
config var N_2 = numLocales / N_1;
if numLocales == 1 {
  N_1 = 1;
  N_2 = 1;
}

const matrixDom : domain(2) dmapped Block(boundingBox = matrixSpace) = matrixSpace;

var A = Matrix(matrixDom, dtype),
    B = Matrix(matrixDom, dtype),
    C = Matrix(matrixDom, dtype),
    D = Matrix(matrixDom, int);

forall a in D do
  a = a.locale.id;

forall (i,j) in matrixDom {
  A[i,j] = j;
  B[i,j] = j;
  C[i,j] = 0;
}

var t = new Timer();
t.start();
forall loc in Locales {
  on loc {
    
    var localDomainA = A.localSubdomain();
    var localDomainB = B.localSubdomain();
    var localDomainC = C.localSubdomain();

    var subArrayA : [localDomainA.dim(0), A.domain.dim(1)] A.eltType = A[localDomainA.dim(0), A.domain.dim(1)];
    var subArrayB : [B.domain.dim(0), localDomainB.dim(1)] B.eltType = B[B.domain.dim(0) ,localDomainB.dim(1)];
    C[localDomainC] = dot(subArrayA, subArrayB);
  }
}
t.stop();

writeln(t.elapsed());
