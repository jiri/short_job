/*  

Short job 1 MI-PRC, 2019/2020

*/ 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>


struct atom{
float x,y,z,charge; 
//x,y,z, a naboj pro kazdy atom
//x,y,z, and charge info for each atom
};

struct grid_s{
int size_x,size_y,size_z; 
// rozmery gridu (mrizky))
// sizes of grid (in x-,y-,and z- dimension)
float spacing_x,spacing_y,spacing_z; 
// mezibodova vzdalenost v gridu
// distances in grid
float offset_x,offset_y,offset_z;    
// posun gridu
// offsets of grid
float *  pot;   
float *  d_pot;                       
// vypocitany potencial v CPU a GPU pameti
// computed potential in grid points
} grid;

struct atom * atoms;
struct atom * d_atoms;
int no_atoms;
// pocet atomu a pole s jejich parametry v CPU a GPU pameti


static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
      exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


void init() 
{
// setup the grid and atoms
  grid.spacing_x=0.15;
  grid.offset_x=0.5;
  grid.spacing_y=0.08;
  grid.offset_y=-0.4;
  grid.spacing_z=0.22;
  grid.offset_z=0.3;
  for (int na=0; na<no_atoms; na++) {
    atoms[na].x=(na%47)+0.229;
    atoms[na].y=(na%31)-10.29; 
    atoms[na].z=(na%19)+50.311;
    atoms[na].charge=(na%8)+0.5;
}}


float body(float t,int n)
{
  float b;
  if (n<5) return 0.0;
  if (t>6.0) return 0.0;
  b=12.0*1.6/t;
  if (b>18.0) b=18.0;
  return b;
}


// zacatek casti k modifikaci
// beginning of part for modification
// muzete pridat vlastni funkce nebo datove struktury, you can also add new functions or data structures


__global__
void compute(int gsizex, int gsizey, int gsizez, float gsx, float gsy, float gsz, float gox, float goy, float goz, struct atom* atoms, int no_atoms, float* gpot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // if (i >= gsizex || j >= gsizey || k >= gsizez) {
    //     return;
    // }

    extern __shared__ struct atom atomcache[];
    
    int numThreads = blockDim.x * blockDim.y * blockDim.z;
    int threadId = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    gpot[k * gsizex * gsizey + j * gsizex + i] = 0.0f;

    // umisteni bodu, location of grid point
    float x = gsx * (float) i + gox;
    float y = gsy * (float) j + goy;
    float z = gsz * (float) k + goz;

    // // printf("%d\n", threadId);
    float pot = 0.0f;
    int loopcount = 0;
    __shared__ int errcount;
    for (int offset = 0; offset < no_atoms; offset += numThreads) {

        __syncthreads();
        if (offset + threadId < no_atoms) {
            atomcache[threadId] = atoms[offset + threadId];
            atomicAdd(&errcount, 1);
        } else {
            // printf("%d + %d = %d (%d)\n", offset, threadId, offset + threadId, numThreads);
        }
        __syncthreads();

        // float pot = 0.0f;

        // if (i >= gsizex || j >= gsizey || k >= gsizez) {
        //     continue;
        // }
        // pro vsechny atomy, for each atom
        // printf("%d\n", min(numThreads, no_atoms - offset));
        for (int na = 0; na < min(numThreads, no_atoms - offset); na++) {
            loopcount++;
            // if (atomcache[na].charge != atoms[offset + na].charge) {
                // errcount++;
            // }
            float dx = x - atomcache[na].x;
            float dy = y - atomcache[na].y;
            float dz = z - atomcache[na].z;
            float charge = atomcache[na].charge;

            pot += charge / sqrt(dx * dx + dy * dy + dz * dz);
        }
        // printf("%f\n", pot);

        // atomicAdd(&gpot[k * gsizex * gsizey + j * gsizex + i], pot);
    //     // gpot[k * gsizex * gsizey + j * gsizex + i] += pot;
    }
    
    if (i < gsizex && j < gsizey && k >= gsizez) {
    //    continue;
    }
    else {
        atomicAdd(&gpot[k * gsizex * gsizey + j * gsizex + i], pot);
    }

    // // __syncthreads();
    
    // umisteni bodu, location of grid point
    // float x = gsx * (float) i + gox;
    // float y = gsy * (float) j + goy;
    // float z = gsz * (float) k + goz;

    float pot2 = 0.0f;
    int loopcount2 = 0;
    // // pro vsechny atomy, for each atom
    for (int na = 0; na < no_atoms; na++) {
        loopcount2++;
        float dx = x - atoms[na].x;
        float dy = y - atoms[na].y;
        float dz = z - atoms[na].z;
        float charge = atoms[na].charge;

        // atomicAdd(&gpot[(k) * gsizex * gsizey + (j) * gsizex + (i)], charge / sqrt(dx * dx + dy * dy + dz * dz));
        pot2 += charge / sqrt(dx * dx + dy * dy + dz * dz);
    }

    // gpot[k * gsizex * gsizey + j * gsizex + i] = pot2;

    if (pot != pot2)
        printf("%d [%d, %d, %d]\t<%d, %d, %d>\t(%d, %d, %d)\t%d / %d / %d\t-\t%f\t%f\t%c\n", threadId, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, loopcount, loopcount2, errcount, pot, pot2, pot == pot2 ? ' ' : 'X');
}

void c_energy(int gsizex, int gsizey, int gsizez, float gsx, float gsy, float gsz, float gox, float goy, float goz, struct atom* atoms, int no_atoms, float* gpot) {
    int tot = gsizex * gsizey * gsizez;

    dim3 grid((gsizex + 7) / 8, (gsizey + 7) / 8, (gsizez + 7) / 8);
    dim3 block_size(8, 8, 8);

    compute<<<grid, block_size, 512 * sizeof(struct atom)>>>(gsizex, gsizey, gsizez, gsx, gsy, gsz, gox, goy, goz, atoms, no_atoms, gpot);
    // compute<<<1, 1, 512 * sizeof(struct atom)>>>(gsizex, gsizey, gsizez, gsx, gsy, gsz, gox, goy, goz, atoms, no_atoms, gpot);
}


// end of part for modification
// konec casti k modifikaci


 
  


int check(int N,float *correct,int gsizex,int gsizey,int gsizez,float *gpot){
// overeni spravnosti, check the correctness
 float crc[8];
 int si,si2;
 for (int i=0; i<8; i++) crc[i]=0.0; 
 
 for (int i=0; i<grid.size_x; i++) {
for (int j=0; j<grid.size_y; j++) {
for (int k=0; k<grid.size_z; k++) {
float x=gpot[(k)*gsizex*gsizey+(j)*gsizex + (i)];//DATA(i,j,k);
si=(i&1)+(j&1)*2+(k&1)*4;
/*si2=(i&2)^(j&2)^(k&2);
if (si2) crc[si]+=x; 
else     crc[si]-=x;
*/
crc[si]+=x; 
}}}
/*
for (int i=0; i<8; i++) printf("%g,",crc[i]);
printf("\n");

for (int i=0; i<8; i++) printf("%g,%g ",crc[i],correct[N*10+i]);
printf("\n");
*/
  for(int i=0;i<8;i++)
    if (fabs(1.0-crc[i]/correct[N*10+i])>0.06)
    {
      printf("ERROR in CRC!!!!\n");
      return 1;  
    }
 return 0;   
}


int main( void ) {

 clock_t start_time,end_time;
 
 int soucet=0,N,i,j,k,n,m,*pomo,v;
 int ri,rj,rk;
 double delta,s_delta=0.0,timea[16];
 float *mA, *mB,*mX,*mX2,s; 

    
  //int tn[4]={1000,1500,2000,2500};  
  float correct[50]={128619,128714,128630,128725,129043,129139,129054,129150, 0, 0,
1.2849e+06,1.28585e+06,1.28501e+06,1.28596e+06,1.28913e+06,1.29009e+06,1.28924e+06,1.2902e+06, 0, 0,
1.285e+08,1.28594e+08,1.28511e+08,1.28605e+08,1.28917e+08,1.29012e+08,1.28929e+08,1.29024e+08, 0, 0,
2.09323e+08,2.09448e+08,2.09287e+08,2.09413e+08,2.10481e+08,2.10609e+08,2.10445e+08,2.10573e+08, 0, 0,
2.31853e+08,2.32026e+08,2.31867e+08,2.3204e+08,2.327e+08,2.32875e+08,2.32715e+08,2.3289e+08, 0, 0};
  int tgx[5]={20,20,20,200,64};
  int tgy[5]={20,20,20,200,64};
  int tgz[5]={20,20,20,200,64};
  int ta[5]={2000,20000,2000000,2000,100000};
  // 16*10^8,81*10^8,64*10^8,52*10^8
  srand (time(NULL));   
  pomo=(int *)malloc(32*1024*1024);    
  v=0;    
  
  for(N=0;N<16;N++) timea[N]=0.0;
  float s_t=0.0;
  for(N=0;N<5;N++)
  {
  grid.size_x=tgx[N];
  grid.size_y=tgy[N];
  grid.size_z=tgz[N];
  no_atoms=ta[N];

 atoms=(struct atom *)malloc(no_atoms * sizeof(struct atom));
 HANDLE_ERROR(cudaMalloc(&d_atoms, no_atoms * sizeof(struct atom)));
  if ((atoms==NULL)||(d_atoms==NULL))
  {
    printf("Alloc error\n");
    return 0;
  }  
  grid.pot=(float *)malloc(grid.size_x * grid.size_y * grid.size_z * sizeof(float));
 HANDLE_ERROR(cudaMalloc(&grid.d_pot, grid.size_x * grid.size_y * grid.size_z * sizeof(float)));
  if ((grid.pot==NULL)||(grid.d_pot==NULL))
  {
    printf("Alloc error\n");
    return 0;
  }  
  
  init();
  HANDLE_ERROR(cudaMemcpy(d_atoms, atoms, no_atoms * sizeof(struct atom), cudaMemcpyHostToDevice));

  //soucet+=vyprazdni(pomo,v);
  start_time=clock();
  // improve performance of this call
  // vylepsit vykonnost tohoto volani
  

  c_energy(grid.size_x,grid.size_y,grid.size_z,grid.spacing_x,grid.spacing_y,grid.spacing_z,grid.offset_x,grid.offset_y,grid.offset_z,d_atoms,no_atoms,grid.d_pot);

  cudaDeviceSynchronize();                        
  end_time=clock();
  HANDLE_ERROR(cudaMemcpy(grid.pot, grid.d_pot, grid.size_x * grid.size_y * grid.size_z * sizeof(float), cudaMemcpyDeviceToHost));
  delta=((double)(end_time-start_time))/CLOCKS_PER_SEC;
  
  timea[N]=delta;
  s_t+=delta;
  rj=check(N,correct,grid.size_x,grid.size_y,grid.size_z,grid.pot);   
  
  if (rj==1)
  {
     printf("BAD result!\n");
     return 0;
   }
   
free(atoms);
free(grid.pot);

cudaFree(d_atoms);
cudaFree(grid.d_pot);
  
  if (s_t>6.0)
  {
    printf("Time limit (6 seconds) is reached (time=%g s). SJ1 points: 0\n",s_t);
    return 0; 
  }
  } // end of N
  printf("%i\n",soucet); 

  for(N=0;N<5;N++) 
  {
    printf("Time %i=%g",N,timea[N]); 
    if (N>=2)
    {
      delta=11;
      delta*=tgx[N];
      delta*=tgy[N];
      delta*=tgz[N];
      delta*=ta[N];
      delta/=timea[N];
      printf(" Perf=%g",delta); 
    }
    printf("\n");
  } 
  printf("Sum of time=%g\n",s_t);
  printf("SJ1 points:%.2f\n",body(s_t,5));

  return 0;
}

