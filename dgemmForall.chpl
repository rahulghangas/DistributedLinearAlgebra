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
  writeln("Window size      =   ", if windowSize>0 then
      windowSize:string
      else "N/A");
  writeln("Number of iterations =   ", iterations);
  writeln();
}

const refChecksum = (iterations+1) *
    (0.25*order*order*order*(order-1.0)*(order-1.0));

var t = new Timer();

private iter block(indexes, window : int) {
  var lowVal = indexes.low;
  while (true) {
    var highVal = lowVal + window - 1;
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
  const commonDim = A.domain.dim(1);
  for niter in 0..iterations {
    coforall loc in A.targetLocales() with (ref t) {
      if loc.id==0 && niter==1 then t.start();
      on loc {
        const localDomainC = C.localSubdomain();
        const (localDim0, localDim1) = localDomainC.dims();
        const windowRange = 0..#windowSize;
        var subArrayA : [localDim0, windowRange] dtype;
        var subArrayB : [windowRange, localDim1] dtype;

        for subArrayChunk in block(commonDim, windowSize) {
          var chunkSize = subArrayChunk.size;

          forall i in localDim0 {
            forall (j, subJ) in zip(subArrayChunk, 0..) {
              subArrayA[i, subJ] = A[i, j];
            }
          }
          forall (i, subI) in zip(subArrayChunk, 0..) {
            forall j in localDim1 {
              subArrayB[subI, j] = B[i, j];
            }
          }

          if chunkSize < windowSize {
            var rest = windowRange#-(windowSize-chunkSize);
            subArrayA[localDim0, rest] = 0;
            subArrayB[rest, localDim1] = 0;
          }

          C.localSlice(localDomainC) += dot(subArrayA, subArrayB);
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
}