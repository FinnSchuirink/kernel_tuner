#ifndef TILE_M
    #define TILE_M 4
#endif
#ifndef TILE_N
    #define TILE_N 4
#endif
#ifndef TILE_K
    #define TILE_K 8
#endif
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 16
#endif
#ifndef VECTOR_WIDTH
    #define VECTOR_WIDTH 1
#endif

__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{

    int row_base = (blockIdx.y * BLOCK_SIZE + threadIdx.y) * TILE_M;
    int col_base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * TILE_N;


    float acc[TILE_M][TILE_N];
    #pragma unroll
    for (int ti = 0; ti < TILE_M; ti++)
        #pragma unroll
        for (int tj = 0; tj < TILE_N; tj++)
            acc[ti][tj] = 0.0f;

    __shared__ float sA[BLOCK_SIZE * TILE_M][TILE_K];
    __shared__ float sB[TILE_K][BLOCK_SIZE * TILE_N];

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int k_off = t * TILE_K;

        #pragma unroll
        for (int ti = 0; ti < TILE_M; ti++) {
            int row = row_base + ti;
            #pragma unroll
            for (int tk = 0; tk < TILE_K; tk++) {
                int col = k_off + tk;
                sA[threadIdx.y * TILE_M + ti][tk] =
                    (row < M && col < K) ? A[row * K + col] : 0.0f;
            }
        }

        #pragma unroll
        for (int tk = 0; tk < TILE_K; tk++) {
            int row = k_off + tk;
            #pragma unroll
            for (int tj = 0; tj < TILE_N; tj++) {
                int col = col_base + tj;
                sB[tk][threadIdx.x * TILE_N + tj] =
                    (row < K && col < N) ? B[row * N + col] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; tk++) {
            #pragma unroll
            for (int ti = 0; ti < TILE_M; ti++) {
                float a_val = sA[threadIdx.y * TILE_M + ti][tk];
                #pragma unroll
                for (int tj = 0; tj < TILE_N; tj++) {
                    acc[ti][tj] += a_val * sB[tk][threadIdx.x * TILE_N + tj];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int ti = 0; ti < TILE_M; ti++) {
        int row = row_base + ti;
        #pragma unroll
        for (int tj = 0; tj < TILE_N; tj++) {
            int col = col_base + tj;
            if (row < M && col < N) {
                int idx = row * N + col;
                C[idx] = alpha * acc[ti][tj] + beta * C[idx];
            }
        }
    }
}
