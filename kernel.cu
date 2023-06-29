#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Tamanho dos vetores, ajustar conforme memoria disponivel
    int n = 268435456;

    int *hostMatrizA, *hostMatrizB, *hostMatrizC, * hostMatrizD;

    int* gpuMatrizA, * gpuMatrizB, * gpuMatrizC;

    hostMatrizA = (int*)malloc(n * sizeof(int));
    hostMatrizB = (int*)malloc(n * sizeof(int));
    hostMatrizC = (int*)malloc(n * sizeof(int));
    hostMatrizD = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        hostMatrizA[i] = i;
        hostMatrizB[i] = i;
        hostMatrizD[i] = hostMatrizA[i] + hostMatrizB[i];
    }

    // Alterar para 1 caso queira usar GPU
    int usarGPU = 0;

    if (usarGPU) {
        // Alocar memoria na GPU
        cudaMalloc((void**)&gpuMatrizA, n * sizeof(int));
        cudaMalloc((void**)&gpuMatrizB, n * sizeof(int));
        cudaMalloc((void**)&gpuMatrizC, n * sizeof(int));

        // Copiando vetores do Host para a GPU
        cudaMemcpy(gpuMatrizA, hostMatrizA, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuMatrizB, hostMatrizB, n * sizeof(int), cudaMemcpyHostToDevice);
        
        int minGrid;
        int tamBloco;

        // Calcula o numero de blocos a serem alocados na GPU
        cudaOccupancyMaxPotentialBlockSize(&minGrid, &tamBloco, vectorAdd, 0, n);
        int tamGrid = (n + tamBloco - 1) / tamBloco;

        // Executar a soma na GPU
        vectorAdd << <tamGrid, tamBloco >> > (gpuMatrizA, gpuMatrizB, gpuMatrizC, n);

        // Copiar o vetor novamente para o Host
        cudaMemcpy(hostMatrizC, gpuMatrizC, n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    else {
            // Soma na CPU
            for (int i = 0; i < n; i++) {
                hostMatrizC[i] = (hostMatrizA[i] + hostMatrizB[i]);
             }
    }

    
    printf("n = %d\n", n);

    //Verifica se o resultado está OK
    for (int i = 0; i < n; i++) {
        if (hostMatrizC[i] != hostMatrizD[i]) {
            printf("Erro na posicao %d\n", i);
        }
    }
    printf("Calculo OK");


    if (usarGPU) {
        cudaFree(gpuMatrizA);
        cudaFree(gpuMatrizB);
        cudaFree(gpuMatrizC);
    }

    free(hostMatrizA);
    free(hostMatrizB);
    free(hostMatrizC);

    return 0;
}
