use Time;
use BlockDist;
use LinearAlgebra;

config type dtype = real;

config param useBlockDist = false;

config const order = 10,
             epsilon = 1e-8,
             iterations = 100,
             windowSize = 10,
             debug = false,
             validate = true,
             correctness = false; // being run in start_test

const vecRange = 0..#order;
const matrixSpace = {vecRange, vecRange};

const matrixDom = matrixSpace dmapped if useBlockDist then
                      new dmap(new Block(boundingBox=matrixSpace)) else
                      defaultDist;

var A = Matrix(matrixDom, dtype),
    B = Matrix(matrixDom, dtype),
    C = Matrix(matrixDom, dtype);

forall (i,j) in matrixDom {
  A[i,j] = j;
  B[i,j] = j;
  C[i,j] = 0;
}

const nTasksPerLocale = here.maxTaskPar;

if !correctness {
  writeln("Chapel Dense matrix-matrix multiplication");
  writeln("Max parallelism      =   ", nTasksPerLocale);
  writeln("Matrix order         =   ", order);
  writeln("Window size          =   ", if windowSize>0 then 
      windowSize:string
      else "N/A");
  writeln("Number of iterations =   ", iterations);
  writeln();
}

const refChecksum = (iterations+1) *
    (0.25*order*order*order*(order-1.0)*(order-1.0));

var t = new Timer();

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

if windowSize == 0 {
  for niter in 0..iterations {
    if niter==1 then t.start();

    forall (i,j) in matrixSpace do
      for k in vecRange do
        C[i,j] += A[i,k] * B[k,j];

  }
  t.stop();
}
else {
  coforall loc in A.targetLocales() with (ref t) {
    on loc {
      var localDomainA = A.localSubdomain();
      var localDomainB = B.localSubdomain();
      var localDomainC = C.localSubdomain();

      const commonDimension = A.domain.dim(1);

      for niter in 0..iterations {
        if here.id==0 && niter==1 then t.start();
        for subArrayAChunk in block(A.domain.dim(1), windowSize) {
          var subArrayA : [localDomainA.dim(0), subArrayAChunk] A.eltType 
                        = A[localDomainA.dim(0), subArrayAChunk];
          var subArrayB : [subArrayAChunk, localDomainB.dim(1)] B.eltType 
                        = B[subArrayAChunk ,localDomainB.dim(1)];
          C[localDomainC] += dot(subArrayA, subArrayB);
        }
      }
    }
  }
  t.stop();
}

if validate {
  const checksum = + reduce C;
  if abs(checksum-refChecksum)/refChecksum > epsilon then
    halt("VALIDATION FAILED! Reference checksum = ", refChecksum,
                           " Checksum = ", checksum);
  else
    writeln("Validation successful");
}

if !correctness {
  const nflops = 2.0*(order**3);
  const avgTime = t.elapsed()/iterations;
  writeln("Rate(MFlop/s) = ", 1e-6*nflops/avgTime, " Time : ", avgTime);
  writeln("Number of locales: ", A.targetLocales().size);
  writeln("\n");
}
