/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
#include <kernels.h>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>

#define TICK()                                                                                                         \
    {                                                                                                                  \
        if (print_kernel_timing) {                                                                                     \
            host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);                                           \
            clock_get_time(cclock, &tick_clockData);                                                                   \
            mach_port_deallocate(mach_task_self(), cclock);                                                            \
        }                                                                                                              \
    }

#define TOCK(str, size)                                                                                                \
    {                                                                                                                  \
        if (print_kernel_timing) {                                                                                     \
            host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);                                           \
            clock_get_time(cclock, &tock_clockData);                                                                   \
            mach_port_deallocate(mach_task_self(), cclock);                                                            \
            std::cerr << str << " ";                                                                                   \
            if ((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec)) \
                std::cerr << tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);       \
            std::cerr << ((tock_clockData.tv_nsec - tick_clockData.tv_nsec) +                                          \
                          ((tock_clockData.tv_nsec < tick_clockData.tv_nsec) ? 1000000000 : 0))                        \
                      << " " << size << std::endl;                                                                     \
        }                                                                                                              \
    }
#else

#define TICK()                                                                                                         \
    {                                                                                                                  \
        if (print_kernel_timing) {                                                                                     \
            clock_gettime(CLOCK_MONOTONIC, &tick_clockData);                                                           \
        }                                                                                                              \
    }

#define TOCK(str, size)                                                                                                \
    {                                                                                                                  \
        if (print_kernel_timing) {                                                                                     \
            clock_gettime(CLOCK_MONOTONIC, &tock_clockData);                                                           \
            std::cerr << str << " ";                                                                                   \
            if ((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec)) \
                std::cerr << tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);       \
            std::cerr << ((tock_clockData.tv_nsec - tick_clockData.tv_nsec) +                                          \
                          ((tock_clockData.tv_nsec < tick_clockData.tv_nsec) ? 1000000000 : 0))                        \
                      << " " << size << std::endl;                                                                     \
        }                                                                                                              \
    }

#endif

// input once
float* gaussian;

// inter-frame
Volume volume;
Volume volume_ref;
float3* vertex;
float3* normal;

// intra-frame
TrackData* trackingResult;
float* reductionoutput;
float** ScaledDepth;
float* floatDepth;
Matrix4 oldPose;
Matrix4 raycastPose;
float3** inputVertex;
float3** inputNormal;

bool print_kernel_timing = false;
#ifdef __APPLE__
clock_serv_t cclock;
mach_timespec_t tick_clockData;
mach_timespec_t tock_clockData;
#else
struct timespec tick_clockData;
struct timespec tock_clockData;
#endif

void Kfusion::languageSpecificConstructor() {
    if (getenv("KERNEL_TIMINGS"))
        print_kernel_timing = true;

    // internal buffers to initialize
    reductionoutput = (float*)calloc(sizeof(float) * 8 * 32, 1);

    ScaledDepth = (float**)calloc(sizeof(float*) * iterations.size(), 1);
    inputVertex = (float3**)calloc(sizeof(float3*) * iterations.size(), 1);
    inputNormal = (float3**)calloc(sizeof(float3*) * iterations.size(), 1);

    for (unsigned int i = 0; i < iterations.size(); ++i) {
        ScaledDepth[i] = (float*)calloc(sizeof(float) * (computationSize.x * computationSize.y) / (int)pow(2, i), 1);
        inputVertex[i] = (float3*)calloc(sizeof(float3) * (computationSize.x * computationSize.y) / (int)pow(2, i), 1);
        inputNormal[i] = (float3*)calloc(sizeof(float3) * (computationSize.x * computationSize.y) / (int)pow(2, i), 1);
    }

    floatDepth = (float*)calloc(sizeof(float) * computationSize.x * computationSize.y, 1);
    vertex = (float3*)calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);
    normal = (float3*)calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);
    trackingResult = (TrackData*)calloc(sizeof(TrackData) * computationSize.x * computationSize.y, 1);

    // ********* BEGIN : Generate the gaussian *************
    size_t gaussianS = radius * 2 + 1;
    gaussian = (float*)calloc(gaussianS * sizeof(float), 1);
    int x;
    for (unsigned int i = 0; i < gaussianS; i++) {
        x = i - 2;
        gaussian[i] = expf(-(x * x) / (2 * delta * delta));
    }
    // ********* END : Generate the gaussian *************

    volume.init(volumeResolution, volumeDimensions);
    volume_ref.init(volumeResolution, volumeDimensions);
    reset();
}

Kfusion::~Kfusion() {
    free(floatDepth);
    free(trackingResult);

    free(reductionoutput);
    for (unsigned int i = 0; i < iterations.size(); ++i) {
        free(ScaledDepth[i]);
        free(inputVertex[i]);
        free(inputNormal[i]);
    }
    free(ScaledDepth);
    free(inputVertex);
    free(inputNormal);

    free(vertex);
    free(normal);
    free(gaussian);

    volume.release();
    volume_ref.release();
}
void Kfusion::reset() { initVolumeKernel(volume); }
void init(){};
// stub
void clean(){};
// stub

void initVolumeKernel(Volume volume) {
    TICK();
    for (unsigned int x = 0; x < volume.size.x; x++)
        for (unsigned int y = 0; y < volume.size.y; y++) {
            for (unsigned int z = 0; z < volume.size.z; z++) {
                // std::cout <<  x << " " << y << " " << z <<"\n";
                volume.setints(x, y, z, make_float2(1.0f, 0.0f));
            }
        }
    TOCK("initVolumeKernel", volume.size.x * volume.size.y * volume.size.z);
}

void bilateralFilterKernel(float* out, const float* in, uint2 size, const float* gaussian, float e_d, int r) {
    TICK()
    uint y;
    float e_d_squared_2 = e_d * e_d * 2;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < size.y; y++) {
        for (uint x = 0; x < size.x; x++) {
            uint pos = x + y * size.x;
            if (in[pos] == 0) {
                out[pos] = 0;
                continue;
            }

            float sum = 0.0f;
            float t = 0.0f;

            const float center = in[pos];

            for (int i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1), clamp(y + j, 0u, size.y - 1));
                    const float curPix = in[curPos.x + curPos.y * size.x];
                    if (curPix > 0) {
                        const float mod = sq(curPix - center);
                        const float factor = gaussian[i + r] * gaussian[j + r] * expf(-mod / e_d_squared_2);
                        t += factor * curPix;
                        sum += factor;
                    }
                }
            }
            out[pos] = t / sum;
        }
    }
    TOCK("bilateralFilterKernel", size.x * size.y);
}

void depth2vertexKernel(float3* vertex, const float* depth, uint2 imageSize, const Matrix4 invK) {
    TICK();
    unsigned int x, y;
#pragma omp parallel for shared(vertex), private(x, y)
    for (y = 0; y < imageSize.y; y++) {
        for (x = 0; x < imageSize.x; x++) {
            if (depth[x + y * imageSize.x] > 0) {
                vertex[x + y * imageSize.x] = depth[x + y * imageSize.x] * (rotate(invK, make_float3(x, y, 1.f)));
            } else {
                vertex[x + y * imageSize.x] = make_float3(0);
            }
        }
    }
    TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void vertex2normalKernel(float3* out, const float3* in, uint2 imageSize) {
    TICK();
    unsigned int x, y;
#pragma omp parallel for shared(out), private(x, y)
    for (y = 0; y < imageSize.y; y++) {
        for (x = 0; x < imageSize.x; x++) {
            const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
            const uint2 pright = make_uint2(min(x + 1, (int)imageSize.x - 1), y);
            const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
            const uint2 pdown = make_uint2(x, min(y + 1, ((int)imageSize.y) - 1));

            const float3 left = in[pleft.x + imageSize.x * pleft.y];
            const float3 right = in[pright.x + imageSize.x * pright.y];
            const float3 up = in[pup.x + imageSize.x * pup.y];
            const float3 down = in[pdown.x + imageSize.x * pdown.y];

            if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
                out[x + y * imageSize.x].x = KFUSION_INVALID;
                continue;
            }
            const float3 dxv = right - left;
            const float3 dyv = down - up;
            out[x + y * imageSize.x] = normalize(cross(dyv, dxv));  // switched dx and dy to get factor -1
        }
    }
    TOCK("vertex2normalKernel", imageSize.x * imageSize.y);
}

void new_reduce(int blockIndex, float* out, TrackData* J, const uint2 Jsize, const uint2 size) {
    float* sums = out + blockIndex * 32;

    float* jtj = sums + 7;
    float* info = sums + 28;
    for (uint i = 0; i < 32; ++i)
        sums[i] = 0;
    float sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9, sums10, sums11, sums12, sums13, sums14,
        sums15, sums16, sums17, sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25, sums26, sums27, sums28,
        sums29, sums30, sums31;
    sums0 = 0.0f;
    sums1 = 0.0f;
    sums2 = 0.0f;
    sums3 = 0.0f;
    sums4 = 0.0f;
    sums5 = 0.0f;
    sums6 = 0.0f;
    sums7 = 0.0f;
    sums8 = 0.0f;
    sums9 = 0.0f;
    sums10 = 0.0f;
    sums11 = 0.0f;
    sums12 = 0.0f;
    sums13 = 0.0f;
    sums14 = 0.0f;
    sums15 = 0.0f;
    sums16 = 0.0f;
    sums17 = 0.0f;
    sums18 = 0.0f;
    sums19 = 0.0f;
    sums20 = 0.0f;
    sums21 = 0.0f;
    sums22 = 0.0f;
    sums23 = 0.0f;
    sums24 = 0.0f;
    sums25 = 0.0f;
    sums26 = 0.0f;
    sums27 = 0.0f;
    sums28 = 0.0f;
    sums29 = 0.0f;
    sums30 = 0.0f;
    sums31 = 0.0f;
// comment me out to try coarse grain parallelism
#pragma omp parallel for reduction(+:sums0,sums1,sums2,sums3,sums4,sums5,sums6,sums7,sums8,sums9,sums10,sums11,sums12,sums13,sums14,sums15,sums16,sums17,sums18,sums19,sums20,sums21,sums22,sums23,sums24,sums25,sums26,sums27,sums28,sums29,sums30,sums31)
    for (uint y = blockIndex; y < size.y; y += 8) {
        for (uint x = 0; x < size.x; x++) {
            const TrackData& row = J[(x + y * Jsize.x)];  // ...
            if (row.result < 1) {
                // accesses sums[28..31]
                /*(sums+28)[1]*/ sums29 += row.result == -4 ? 1 : 0;
                /*(sums+28)[2]*/ sums30 += row.result == -5 ? 1 : 0;
                /*(sums+28)[3]*/ sums31 += row.result > -4 ? 1 : 0;

                continue;
            }
            // Error part
            /*sums[0]*/ sums0 += row.error * row.error;

            // JTe part
            /*for(int i = 0; i < 6; ++i)
             sums[i+1] += row.error * row.J[i];*/
            sums1 += row.error * row.J[0];
            sums2 += row.error * row.J[1];
            sums3 += row.error * row.J[2];
            sums4 += row.error * row.J[3];
            sums5 += row.error * row.J[4];
            sums6 += row.error * row.J[5];

            // JTJ part, unfortunatly the double loop is not unrolled well...
            /*(sums+7)[0]*/ sums7 += row.J[0] * row.J[0];
            /*(sums+7)[1]*/ sums8 += row.J[0] * row.J[1];
            /*(sums+7)[2]*/ sums9 += row.J[0] * row.J[2];
            /*(sums+7)[3]*/ sums10 += row.J[0] * row.J[3];

            /*(sums+7)[4]*/ sums11 += row.J[0] * row.J[4];
            /*(sums+7)[5]*/ sums12 += row.J[0] * row.J[5];

            /*(sums+7)[6]*/ sums13 += row.J[1] * row.J[1];
            /*(sums+7)[7]*/ sums14 += row.J[1] * row.J[2];
            /*(sums+7)[8]*/ sums15 += row.J[1] * row.J[3];
            /*(sums+7)[9]*/ sums16 += row.J[1] * row.J[4];

            /*(sums+7)[10]*/ sums17 += row.J[1] * row.J[5];

            /*(sums+7)[11]*/ sums18 += row.J[2] * row.J[2];
            /*(sums+7)[12]*/ sums19 += row.J[2] * row.J[3];
            /*(sums+7)[13]*/ sums20 += row.J[2] * row.J[4];
            /*(sums+7)[14]*/ sums21 += row.J[2] * row.J[5];

            /*(sums+7)[15]*/ sums22 += row.J[3] * row.J[3];
            /*(sums+7)[16]*/ sums23 += row.J[3] * row.J[4];
            /*(sums+7)[17]*/ sums24 += row.J[3] * row.J[5];

            /*(sums+7)[18]*/ sums25 += row.J[4] * row.J[4];
            /*(sums+7)[19]*/ sums26 += row.J[4] * row.J[5];

            /*(sums+7)[20]*/ sums27 += row.J[5] * row.J[5];

            // extra info here
            /*(sums+28)[0]*/ sums28 += 1;
        }
    }
    sums[0] = sums0;
    sums[1] = sums1;
    sums[2] = sums2;
    sums[3] = sums3;
    sums[4] = sums4;
    sums[5] = sums5;
    sums[6] = sums6;
    sums[7] = sums7;
    sums[8] = sums8;
    sums[9] = sums9;
    sums[10] = sums10;
    sums[11] = sums11;
    sums[12] = sums12;
    sums[13] = sums13;
    sums[14] = sums14;
    sums[15] = sums15;
    sums[16] = sums16;
    sums[17] = sums17;
    sums[18] = sums18;
    sums[19] = sums19;
    sums[20] = sums20;
    sums[21] = sums21;
    sums[22] = sums22;
    sums[23] = sums23;
    sums[24] = sums24;
    sums[25] = sums25;
    sums[26] = sums26;
    sums[27] = sums27;
    sums[28] = sums28;
    sums[29] = sums29;
    sums[30] = sums30;
    sums[31] = sums31;
}
void reduceKernel(float* out, TrackData* J, const uint2 Jsize, const uint2 size) {
    TICK();
    int blockIndex;
#ifdef OLDREDUCE
#pragma omp parallel for private(blockIndex)
#endif
    for (blockIndex = 0; blockIndex < 8; blockIndex++) {
#ifdef OLDREDUCE
        float S[112][32];  // this is for the final accumulation
        // we have 112 threads in a blockdim
        // and 8 blocks in a gridDim?
        // ie it was launched as <<<8,112>>>
        uint sline;  // threadIndex.x
        float sums[32];

        for (int threadIndex = 0; threadIndex < 112; threadIndex++) {
            sline = threadIndex;
            float* jtj = sums + 7;
            float* info = sums + 28;
            for (uint i = 0; i < 32; ++i)
                sums[i] = 0;

            for (uint y = blockIndex; y < size.y; y += 8 /*gridDim.x*/) {
                for (uint x = sline; x < size.x; x += 112 /*blockDim.x*/) {
                    const TrackData& row = J[(x + y * Jsize.x)];  // ...

                    if (row.result < 1) {
                        // accesses S[threadIndex][28..31]
                        info[1] += row.result == -4 ? 1 : 0;
                        info[2] += row.result == -5 ? 1 : 0;
                        info[3] += row.result > -4 ? 1 : 0;
                        continue;
                    }
                    // Error part
                    sums[0] += row.error * row.error;

                    // JTe part
                    for (int i = 0; i < 6; ++i)
                        sums[i + 1] += row.error * row.J[i];

                    // JTJ part, unfortunatly the double loop is not unrolled well...
                    jtj[0] += row.J[0] * row.J[0];
                    jtj[1] += row.J[0] * row.J[1];
                    jtj[2] += row.J[0] * row.J[2];
                    jtj[3] += row.J[0] * row.J[3];

                    jtj[4] += row.J[0] * row.J[4];
                    jtj[5] += row.J[0] * row.J[5];

                    jtj[6] += row.J[1] * row.J[1];
                    jtj[7] += row.J[1] * row.J[2];
                    jtj[8] += row.J[1] * row.J[3];
                    jtj[9] += row.J[1] * row.J[4];

                    jtj[10] += row.J[1] * row.J[5];

                    jtj[11] += row.J[2] * row.J[2];
                    jtj[12] += row.J[2] * row.J[3];
                    jtj[13] += row.J[2] * row.J[4];
                    jtj[14] += row.J[2] * row.J[5];

                    jtj[15] += row.J[3] * row.J[3];
                    jtj[16] += row.J[3] * row.J[4];
                    jtj[17] += row.J[3] * row.J[5];

                    jtj[18] += row.J[4] * row.J[4];
                    jtj[19] += row.J[4] * row.J[5];

                    jtj[20] += row.J[5] * row.J[5];

                    // extra info here
                    info[0] += 1;
                }
            }

            for (int i = 0; i < 32; ++i) {  // copy over to shared memory
                S[sline][i] = sums[i];
            }
            // WE NO LONGER NEED TO DO THIS AS the threads execute sequentially inside a for loop

        }  // threads now execute as a for loop.
           // so the __syncthreads() is irrelevant

        for (int ssline = 0; ssline < 32;
             ssline++) {  // sum up columns and copy to global memory in the final 32 threads
            for (unsigned i = 1; i < 112 /*blockDim.x*/; ++i) {
                S[0][ssline] += S[i][ssline];
            }
            out[ssline + blockIndex * 32] = S[0][ssline];
        }
#else
        new_reduce(blockIndex, out, J, Jsize, size);
#endif
    }

    TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
    for (int j = 1; j < 8; ++j) {
        values[0] += values[j];
        // std::cerr << "REDUCE ";for(int ii = 0; ii < 32;ii++)
        // std::cerr << values[0][ii] << " ";
        // std::cerr << "\n";
    }
    TOCK("reduceKernel", 512);
}

void trackKernel(TrackData* output, const float3* inVertex, const float3* inNormal, uint2 inSize,
                 const float3* refVertex, const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
                 const Matrix4 view, const float dist_threshold, const float normal_threshold) {
    TICK();
    uint2 pixel = make_uint2(0, 0);
    unsigned int pixely, pixelx;
#pragma omp parallel for shared(output), private(pixel, pixelx, pixely)
    for (pixely = 0; pixely < inSize.y; pixely++) {
        for (pixelx = 0; pixelx < inSize.x; pixelx++) {
            pixel.x = pixelx;
            pixel.y = pixely;

            TrackData& row = output[pixel.x + pixel.y * refSize.x];

            if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
                row.result = -1;
                continue;
            }

            const float3 projectedVertex = Ttrack * inVertex[pixel.x + pixel.y * inSize.x];
            const float3 projectedPos = view * projectedVertex;
            const float2 projPixel =
                make_float2(projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);
            if (projPixel.x < 0 || projPixel.x > refSize.x - 1 || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
                row.result = -2;
                continue;
            }

            const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
            const float3 referenceNormal = refNormal[refPixel.x + refPixel.y * refSize.x];

            if (referenceNormal.x == KFUSION_INVALID) {
                row.result = -3;
                continue;
            }

            const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x] - projectedVertex;
            const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

            if (length(diff) > dist_threshold) {
                row.result = -4;
                continue;
            }
            if (dot(projectedNormal, referenceNormal) < normal_threshold) {
                row.result = -5;
                continue;
            }

            // Useable informations are recorded here. .J contains two float3
            row.result = 1;
            row.error = dot(referenceNormal, diff);
            ((float3*)row.J)[0] = referenceNormal;
            ((float3*)row.J)[1] = cross(projectedVertex, referenceNormal);
        }
    }
    TOCK("trackKernel", inSize.x * inSize.y);
}

void mm2metersKernel(float* out, uint2 outSize, const ushort* in, uint2 inSize) {
    TICK();
    // Check for unsupported conditions
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }
    if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }

    int ratio = inSize.x / outSize.x;
    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < outSize.y; y++)
        for (unsigned int x = 0; x < outSize.x; x++) {
            out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio] / 1000.0f;
        }
    TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize, const float e_d, const int r) {
    TICK();
    uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < outSize.y; y++) {
        for (unsigned int x = 0; x < outSize.x; x++) {
            uint2 pixel = make_uint2(x, y);
            const uint2 centerPixel = 2 * pixel;

            float sum = 0.0f;
            float t = 0.0f;
            const float center = in[centerPixel.x + centerPixel.y * inSize.x];
            for (int i = -r + 1; i <= r; ++i) {
                for (int j = -r + 1; j <= r; ++j) {
                    uint2 cur = make_uint2(clamp(make_int2(centerPixel.x + j, centerPixel.y + i), make_int2(0),
                                                 make_int2(2 * outSize.x - 1, 2 * outSize.y - 1)));
                    float current = in[cur.x + cur.y * inSize.x];
                    if (fabsf(current - center) < e_d) {
                        sum += 1.0f;
                        t += current;
                    }
                }
            }
            out[pixel.x + pixel.y * outSize.x] = t / sum;
        }
    }
    TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}

void integrateKernel(Volume vol, const float* depth, uint2 depthSize, const Matrix4 invTrack, const Matrix4 K,
                     const float mu, const float maxweight) {
    TICK();
    const float3 delta = rotate(invTrack, make_float3(0, 0, vol.dim.z / vol.size.z));
    const float3 cameraDelta = rotate(K, delta);
    unsigned int y;
#pragma omp parallel for shared(vol), private(y)
    for (y = 0; y < vol.size.y; y++)
        for (unsigned int x = 0; x < vol.size.x; x++) {
            uint3 pix = make_uint3(x, y, 0);  // pix.x = x;pix.y = y;
            float3 pos = invTrack * vol.pos(pix);
            float3 cameraX = K * pos;

            for (pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX += cameraDelta) {
                if (pos.z < 0.0001f)  // some near plane constraint
                    continue;
                const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f, cameraX.y / cameraX.z + 0.5f);
                if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0 || pixel.y > depthSize.y - 1)
                    continue;
                const uint2 px = make_uint2(pixel.x, pixel.y);
                if (depth[px.x + px.y * depthSize.x] == 0)
                    continue;
                const float diff = (depth[px.x + px.y * depthSize.x] - cameraX.z) *
                                   std::sqrt(1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));
                if (diff > -mu) {
                    const float sdf = fminf(1.f, diff / mu);
                    float2 data = vol[pix];
                    // we update the volume here using TSDF
                    data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
                    data.y = fminf(data.y + 1, maxweight);
                    vol.set(pix, data);
                }
            }
        }
    TOCK("integrateKernel", vol.size.x * vol.size.y);
}
float4 raycast(const Volume volume, const uint2 pos, const Matrix4 view, const float nearPlane, const float farPlane,
               const float step, const float largestep) {
    const float3 origin = get_translation(view);
    const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

    // intersect ray with a box
    // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
    // compute intersection of ray with all six bbox planes
    const float3 invR = make_float3(1.0f) / direction;
    const float3 tbot = -1 * invR * origin;
    const float3 ttop = invR * (volume.dim - origin);

    // re-order intersections to find smallest and largest on each axis
    const float3 tmin = fminf(ttop, tbot);
    const float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    const float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    // check against near and far plane
    const float tnear = fmaxf(largest_tmin, nearPlane);
    const float tfar = fminf(smallest_tmax, farPlane);

    if (tnear < tfar) {
        // first walk with largesteps until we found a hit
        float t = tnear;
        float stepsize = largestep;
        float f_t = volume.interp(origin + direction * t);
        float f_tt = 0;
        if (f_t > 0) {  // ups, if we were already in it, then don't render anything here
            for (; t < tfar; t += stepsize) {
                f_tt = volume.interp(origin + direction * t);
                if (f_tt < 0)  // got it, jump out of inner loop
                    break;
                if (f_tt < 0.8f)  // coming closer, reduce stepsize
                    stepsize = step;
                f_t = f_tt;
            }
            if (f_tt < 0) {  // got it, calculate accurate intersection
                t = t + stepsize * f_tt / (f_t - f_tt);
                return make_float4(origin + direction * t, t);
            }
        }
    }
    return make_float4(0);
}
void raycastKernel(float3* vertex, float3* normal, uint2 inputSize, const Volume integration, const Matrix4 view,
                   const float nearPlane, const float farPlane, const float step, const float largestep) {
    TICK();
    unsigned int y;
#pragma omp parallel for shared(normal, vertex), private(y)
    for (y = 0; y < inputSize.y; y++)
        for (unsigned int x = 0; x < inputSize.x; x++) {
            uint2 pos = make_uint2(x, y);

            const float4 hit = raycast(integration, pos, view, nearPlane, farPlane, step, largestep);
            if (hit.w > 0.0) {
                vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
                float3 surfNorm = integration.grad(make_float3(hit));
                if (length(surfNorm) == 0) {
                    // normal[pos] = normalize(surfNorm); // APN added
                    normal[pos.x + pos.y * inputSize.x].x = KFUSION_INVALID;
                } else {
                    normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
                }
            } else {
                // std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w <<"\n";
                vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
                normal[pos.x + pos.y * inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
            }
        }
    TOCK("raycastKernel", inputSize.x * inputSize.y);
}

bool updatePoseKernel(Matrix4& pose, const float* output, float icp_threshold) {
    bool res = false;
    TICK();
    // Update the pose regarding the tracking result
    // transfer reductionoutput into an 8x32 matrix
    TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
    // take the index 1-27 elemnt out for calculating pose shift, but only
    // the first line. TODO: why only use the first row, there are 7 more
    // lines to use.
    TooN::Vector<6> x = solve(values[0].slice<1, 27>());
    TooN::SE3<> delta(x);
    pose = toMatrix4(delta) * pose;

    // Return if pose need further update
    if (norm(x) < icp_threshold)
        res = true;

    TOCK("updatePoseKernel", 1);
    return res;
}

bool checkPoseKernel(Matrix4& pose, Matrix4 oldPose, const float* output, uint2 imageSize, float track_threshold) {
    // Check the tracking result, and go back to the previous camera position if necessary

    TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
    // UT:
    // printf("    value(0,0) = %f\n", values(0,0));
    // printf("    value(0,28) = %f\n", values(0,28));
    // printf("    ratio = %f\n", std::sqrt(values(0, 0) / values(0, 28)));

    if ((std::sqrt(values(0, 0) / values(0, 28)) > average_matching_error_threshold) ||
        (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
        pose = oldPose;
        return false;
    } else {
        return true;
    }
}

void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize) {
    TICK();
    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < normalSize.y; y++)
        for (unsigned int x = 0; x < normalSize.x; x++) {
            uint pos = (x + y * normalSize.x);
            float3 n = normal[pos];
            if (n.x == -2) {
                out[pos] = make_uchar3(0, 0, 0);
            } else {
                n = normalize(n);
                out[pos] = make_uchar3(n.x * 128 + 128, n.y * 128 + 128, n.z * 128 + 128);
            }
        }
    TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}

void renderDepthKernel(uchar4* out, float* depth, uint2 depthSize, const float nearPlane, const float farPlane) {
    TICK();

    float rangeScale = 1 / (farPlane - nearPlane);

    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < depthSize.y; y++) {
        int rowOffeset = y * depthSize.x;
        for (unsigned int x = 0; x < depthSize.x; x++) {
            unsigned int pos = rowOffeset + x;

            if (depth[pos] < nearPlane)
                out[pos] = make_uchar4(255, 255, 255, 0);  // The forth value is a padding in order to align memory
            else {
                if (depth[pos] > farPlane)
                    out[pos] = make_uchar4(0, 0, 0, 0);  // The forth value is a padding in order to align memory
                else {
                    const float d = (depth[pos] - nearPlane) * rangeScale;
                    out[pos] = gs2rgb(d);
                }
            }
        }
    }
    TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}

void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize) {
    TICK();

    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < outSize.y; y++)
        for (unsigned int x = 0; x < outSize.x; x++) {
            uint pos = x + y * outSize.x;
            switch (data[pos].result) {
                case 1:
                    out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
                    break;
                case -1:
                    out[pos] = make_uchar4(0, 0, 0, 0);  // no input BLACK
                    break;
                case -2:
                    out[pos] = make_uchar4(255, 0, 0, 0);  // not in image RED
                    break;
                case -3:
                    out[pos] = make_uchar4(0, 255, 0, 0);  // no correspondence GREEN
                    break;
                case -4:
                    out[pos] = make_uchar4(0, 0, 255, 0);  // to far away BLUE
                    break;
                case -5:
                    out[pos] = make_uchar4(255, 255, 0, 0);  // wrong normal YELLOW
                    break;
                default:
                    out[pos] = make_uchar4(255, 128, 128, 0);
                    break;
            }
        }
    TOCK("renderTrackKernel", outSize.x * outSize.y);
}

void renderVolumeKernel(uchar4* out, const uint2 depthSize, const Volume volume, const Matrix4 view,
                        const float nearPlane, const float farPlane, const float step, const float largestep,
                        const float3 light, const float3 ambient) {
    TICK();
    unsigned int y;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < depthSize.y; y++) {
        for (unsigned int x = 0; x < depthSize.x; x++) {
            const uint pos = x + y * depthSize.x;

            float4 hit = raycast(volume, make_uint2(x, y), view, nearPlane, farPlane, step, largestep);
            if (hit.w > 0) {
                const float3 test = make_float3(hit);
                const float3 surfNorm = volume.grad(test);
                if (length(surfNorm) > 0) {
                    const float3 diff = normalize(light - test);
                    const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                    const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                    out[pos] = make_uchar4(col.x, col.y, col.z, 0);  // The forth value is a padding to align memory
                } else {
                    out[pos] = make_uchar4(0, 0, 0, 0);  // The forth value is a padding to align memory
                }
            } else {
                out[pos] = make_uchar4(0, 0, 0, 0);  // The forth value is a padding to align memory
            }
        }
    }
    TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

void compareVolumeKernel(uchar4* out, const uint2 depthSize, const Volume volume, const Matrix4 view,
                        const float nearPlane, const float farPlane, const float step, const float largestep,
                        const float3 light, const float3 ambient, const char* ref_file) {
    TICK();
    std::ifstream fin(ref_file);
    uint32_t i, j ,k;
    float tmp;
    for (i = 0; i < volume.size.x; i++) {
        for (j = 0; j < volume.size.y; j++) {
            for (k = 0; k < volume.size.z; k++) {
                uint3 voxel_index = make_uint3(i, j, k);
                fin >> tmp;
                volume_ref.set(voxel_index, make_float2(tmp, volume[make_uint3(i,j,k)].y));
            }
        }
    }
    fin.close();

    unsigned int y;
    const float eps_high = 0.01;
    const float eps_low = 0.004;
    const uint32_t dim = 48;
#pragma omp parallel for shared(out), private(y)
    for (y = 0; y < depthSize.y; y++) {
        for (unsigned int x = 0; x < depthSize.x; x++) {
            const uint pos = x + y * depthSize.x;

            float4 hit = raycast(volume, make_uint2(x, y), view, nearPlane, farPlane, step, largestep);
            float4 hit_ref = raycast(volume_ref, make_uint2(x, y), view, nearPlane, farPlane, step, largestep);
            if (hit.w > 0) {
                const float3 test = make_float3(hit);
                const float3 test_ref = make_float3(hit_ref);
                const float3 surfNorm = volume.grad(test);
                if (length(surfNorm) > 0) {
                    const float3 diff = normalize(light - test);
                    const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                    const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                    out[pos] = make_uchar4(col.x, col.y, col.z, 0);  // The forth value is a padding to align memory
                    if (test_ref.x > test.x + eps_low || test_ref.x < test.x - eps_low
                       || test_ref.y > test.y + eps_low || test_ref.y < test.y - eps_low
                       || test_ref.z > test.z + eps_low || test_ref.z < test.z - eps_low){
                        if (col.x > dim) {
                            out[pos].y /= 1.25;
                            out[pos].z /= 1.25;
                            if (test_ref.x > test.x + eps_high || test_ref.x < test.x - eps_high
                               || test_ref.y > test.y + eps_high || test_ref.y < test.y - eps_high
                               || test_ref.z > test.z + eps_high || test_ref.z < test.z - eps_high){
                                out[pos].x = 255;
                            } else {
                                float avg_ref = (test_ref.x + test_ref.y + test_ref.z)/3.0;
                                float avg = (test.x + test.y + test.z)/3.0;
                                out[pos].x += (255 - out[pos].x)*(avg_ref - avg - eps_low)/(eps_high - eps_low);
                            }
                        }
                    }
                } else {
                    out[pos] = make_uchar4(0, 0, 0, 0);  // The forth value is a padding to align memory
                }
            } else {
                out[pos] = make_uchar4(0, 0, 0, 0);  // The forth value is a padding to align memory
            }
        }
    }
    TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

bool Kfusion::preprocessing(const ushort* inputDepth, const uint2 inputSize) {
    mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
    bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize, gaussian, e_delta, radius);

    return true;
}

// Estimate the new pose based on the comparison between local vertex, normal
// and global vertex and normal.
bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {
    if (frame % tracking_rate != 0)
        return false;

    // half sample the input depth maps into the pyramid levels
    // iteration.size = 3, corresponding to 3 pyramid level knobs
    for (unsigned int i = 1; i < iterations.size(); ++i) {
        halfSampleRobustImageKernel(
            ScaledDepth[i], ScaledDepth[i - 1],
            make_uint2(computationSize.x / (int)pow(2, i - 1), computationSize.y / (int)pow(2, i - 1)), e_delta * 3, 1);
    }

    // prepare the 3D information from the input depth maps
    // Including local vertex and normal information from alignment
    uint2 localimagesize = computationSize;
    for (unsigned int i = 0; i < iterations.size(); ++i) {
        Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
        depth2vertexKernel(inputVertex[i], ScaledDepth[i], localimagesize, invK);
        vertex2normalKernel(inputNormal[i], inputVertex[i], localimagesize);
        localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);
    }

    oldPose = pose;
    const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

    // UT:
    // printf("frame = %d, csr = %d, icp = %f, pose = (%f, %f, %f), err = %f\n", frame,
    //         640/computationSize.x, icp_threshold, pose.data[0].w, pose.data[1].w,
    //         pose.data[2].w, current_matching_error);

    // The ICP start from the most coarse level
    for (int level = iterations.size() - 1; level >= 0; --level) {
        // Computing the local frame size for the current pyramid size
        uint2 localimagesize =
            make_uint2(computationSize.x / (int)pow(2, level), computationSize.y / (int)pow(2, level));
        for (int i = 0; i < iterations[level]; ++i) {
            // printf("frame = %d, iterations = %d, i = %d\n", frame, iterations[level], i);
            // Tracking result mainly records the difference bewteen local
            // normal info with global normal info, as well as all the pixels
            // that are invalid
            trackKernel(trackingResult, inputVertex[level], inputNormal[level], localimagesize, vertex, normal,
                        computationSize, pose, projectReference, dist_threshold, normal_threshold);

            // ReduceKernel analyzes the trackingResult and translate it a 8x32
            // information matrix
            reduceKernel(reductionoutput, trackingResult, computationSize, localimagesize);

            // UdatePoseKernel analyzes the reductionoutput and translate it
            // into pose change. It will keep updating the pose until the icp
            // is met
            if (updatePoseKernel(pose, reductionoutput, icp_threshold))
                break;
        }
    }

    TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(reductionoutput);
    // UT:
    // printf("    value(0,0) = %f\n", values(0,0));
    // printf("    value(0,28) = %f\n", values(0,28));
    current_matching_error = (std::sqrt(values(0, 0)) * 10000) / (values(0, 28));
    // current_matching_error = (std::sqrt(values(0, 0) / values(0, 28))

    // Determine whether the updated pose should be reversed.
    return checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {
    bool doRaycast = false;

    if (frame > 2) {
        raycastPose = pose;
        raycastKernel(vertex, normal, computationSize, volume, raycastPose * getInverseCameraMatrix(k), nearPlane,
                      farPlane, step, 0.75f * mu);
    }

    return doRaycast;
}

// update volume using the new pose and the current depth information
bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {
    bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);

    if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
        // TODO: why using floatDepth instead of ScaleDepth[0]?
        integrateKernel(volume, floatDepth, computationSize, inverse(pose), getCameraMatrix(k), mu, maxweight);
        doIntegrate = true;
    } else {
        doIntegrate = false;
    }

    return doIntegrate;
}


// UT: Feature detection.
bool Kfusion::checkFeature(uint frame) {
    return checkFeatureKernel(floatDepth, computationSize);
}

void Kfusion::dumpVolume(const char* filename) {
    std::ofstream fDumpFile;

    if (filename == NULL) {
        return;
    }

    std::cout << "Dumping the volumetric representation on file: " << filename << std::endl;
    //fDumpFile.open(filename, std::ios::out | std::ios::binary);
    fDumpFile.open(filename);
    if (fDumpFile.fail()) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    // Dump on file without the y component of the short2 variable
    // for (unsigned int i = 0; i < volume.size.x * volume.size.y * volume.size.z; i++) {
    //     fDumpFile.write((char*)(volume.data + i), sizeof(short));
    // }

    int i, j, k;
    for (i = 0; i < volume.size.x; i++) {
        for (j = 0; j < volume.size.y; j++) {
            for (k = 0; k < volume.size.z; k++) {
                fDumpFile << volume[make_uint3(i,j,k)].x << ' ';
            }
        }
    }
    fDumpFile.close();
}

void Kfusion::renderVolume(uchar4* out, uint2 outputSize, int frame, int raycast_rendering_rate, float4 k,
                           float largestep) {
    if (frame % raycast_rendering_rate == 0)
        renderVolumeKernel(out, outputSize, volume, *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
                           farPlane * 2.0f, step, largestep, light, ambient);
}

void Kfusion::compareVolume(uchar4* out, uint2 outputSize, int frame, int raycast_rendering_rate, float4 k,
                           float largestep, const char* ref_file) {
    if (frame % raycast_rendering_rate == 0)
        compareVolumeKernel(out, outputSize, volume, *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
                           farPlane * 2.0f, step, largestep, light, ambient, ref_file);
}

void Kfusion::renderTrack(uchar4* out, uint2 outputSize) { renderTrackKernel(out, trackingResult, outputSize); }

void Kfusion::renderDepth(uchar4* out, uint2 outputSize) {
    renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
}

void Kfusion::computeFrame(const ushort* inputDepth, const uint2 inputSize, float4 k, uint integration_rate,
                           uint tracking_rate, float icp_threshold, float mu, const uint frame) {
    preprocessing(inputDepth, inputSize);
    _tracked = tracking(k, icp_threshold, tracking_rate, frame);
    _integrated = integration(k, integration_rate, mu, frame);
    raycasting(k, mu, frame);
}

void synchroniseDevices() {
    // Nothing to do in the C++ implementation
}

// UT:
void Kfusion::updateComputationSize(uint2 cSize) {
    uint2 old_cSize = computationSize;
    computationSize = make_uint2(cSize.x, cSize.y);

    if (old_cSize.x == cSize.x) {
        return;
    }

    float3* new_vertex = (float3*)calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);
    float3* new_normal = (float3*)calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);

    int ratio = 1;
    int i, j, k, l;
    if (cSize.x > old_cSize.x) {  // expanding
        ratio = cSize.x / old_cSize.x;
        for (i = 0; i < old_cSize.x; i++) {
            for (j = 0; j < old_cSize.y; j++) {
                for (k = 0; k < ratio; k++) {
                    for (l = 0; l < ratio; l++) {
                        new_vertex[(j * ratio + l) * cSize.x + ratio * i + k] = vertex[j * old_cSize.x + i];
                        new_normal[(j * ratio + l) * cSize.x + ratio * i + k] = normal[j * old_cSize.x + i];
                    }
                }
            }
        }

    } else if (cSize.x < old_cSize.x) {  // compressing
        ratio = old_cSize.x / cSize.x;
        for (i = 0; i < cSize.x; i++) {
            for (j = 0; j < cSize.y; j++) {
                float3 sum_vertex = make_float3(0.0, 0.0, 0.0);
                float3 sum_normal = make_float3(0.0, 0.0, 0.0);
                for (k = 0; k < ratio; k++) {
                    for (l = 0; l < ratio; l++) {
                        sum_vertex += vertex[(j * ratio + l) * old_cSize.x + ratio * i + k];
                        sum_normal += normal[(j * ratio + l) * old_cSize.x + ratio * i + k];
                    }
                }
                new_vertex[j * cSize.x + i] = sum_vertex / (ratio * ratio);
                new_normal[j * cSize.x + i] = sum_normal / (ratio * ratio);
            }
        }
    }

    delete[] vertex;
    delete[] normal;

    vertex = new_vertex;
    normal = new_normal;

    return;
}

/* Returns true if the intra-frame deviation is small.
 * 5 regions with proportional number of points
bool checkFeatureKernel(float* dep, uint2 size) {

    // Features are not needed when the resolution is the highest
    if (computationSize.x == 640 && computationSize.y == 480) {
        return false;
    }

    // The sobel filter may be used here to detect edges
    int top_blank = size.y * 0.1;
    int left_blank = top_blank;
    int right_blank = top_blank;

    float2 vrange_tl = make_float2(0.2, 0.4);
    float2 hrange_tl = make_float2(0.2, 0.4);

    float2 vrange_tr = make_float2(0.6, 0.8);
    float2 hrange_tr = make_float2(0.2, 0.4);

    float2 vrange_bl = make_float2(0.2, 0.4);
    float2 hrange_bl = make_float2(0.6, 0.8);

    float2 vrange_br = make_float2(0.6, 0.8);
    float2 hrange_br = make_float2(0.6, 0.8);

    float2 vrange_mm = make_float2(0.4, 0.6);
    float2 hrange_mm = make_float2(0.4, 0.6);

    float avg_tl = depth_avg(dep, size, vrange_tl, hrange_tl);
    float avg_tr = depth_avg(dep, size, vrange_tr, hrange_tr);
    float avg_bl = depth_avg(dep, size, vrange_bl, hrange_bl);
    float avg_br = depth_avg(dep, size, vrange_br, hrange_br);
    float avg_mm = depth_avg(dep, size, vrange_mm, hrange_mm);

    // printf("avg_tl=%f, avg_tr=%f, avg_bl=%f, avg_br=%f, avg_mm=%f\n",
    //                      avg_tl, avg_tr, avg_bl, avg_br, avg_mm);

    float std_tl = depth_std(avg_tl, dep, size, vrange_tl, hrange_tl);
    float std_tr = depth_std(avg_tr, dep, size, vrange_tr, hrange_tr);
    float std_bl = depth_std(avg_bl, dep, size, vrange_bl, hrange_bl);
    float std_br = depth_std(avg_br, dep, size, vrange_br, hrange_br);
    float std_mm = depth_std(avg_mm, dep, size, vrange_mm, hrange_mm);

    // printf("std_tl=%f, std_tr=%f, std_bl=%f, std_br=%f, std_mm=%f\n",
    //                         std_tl, std_tr, std_bl, std_br, std_mm);

    float std_thres = 0.12;
    if (std_tl < std_thres && std_tr < std_thres && std_bl < std_thres && std_br < std_thres && std_mm < std_thres) {
        return true;
    } else {
        return false;
    }
}
*/

/* Returns true if the intra-frame deviation is small.
 * 4 regions with fixed number of points
 */
bool checkFeatureKernel(float* dep, uint2 size) {

    // Features always check since number of points is fixed
    // if (computationSize.x == 640 && computationSize.y == 480) {
    //     return false;
    // }

    // The sobel filter may be used here to detect edges
    int blank_ratio = 0.1;

    float2 vrange_tl = make_float2(0.1, 0.5);
    float2 hrange_tl = make_float2(0.1, 0.5);

    float2 vrange_tr = make_float2(0.5, 0.9);
    float2 hrange_tr = make_float2(0.1, 0.5);

    float2 vrange_bl = make_float2(0.1, 0.5);
    float2 hrange_bl = make_float2(0.5, 0.9);

    float2 vrange_br = make_float2(0.5, 0.9);
    float2 hrange_br = make_float2(0.5, 0.9);

    float avg_tl = depth_avg(dep, size, vrange_tl, hrange_tl);
    float avg_tr = depth_avg(dep, size, vrange_tr, hrange_tr);
    float avg_bl = depth_avg(dep, size, vrange_bl, hrange_bl);
    float avg_br = depth_avg(dep, size, vrange_br, hrange_br);

    // printf("avg_tl=%f, avg_tr=%f, avg_bl=%f, avg_br=%f, avg_mm=%f\n",
    //                      avg_tl, avg_tr, avg_bl, avg_br, avg_mm);

    float std_tl = depth_std(avg_tl, dep, size, vrange_tl, hrange_tl);
    float std_tr = depth_std(avg_tr, dep, size, vrange_tr, hrange_tr);
    float std_bl = depth_std(avg_bl, dep, size, vrange_bl, hrange_bl);
    float std_br = depth_std(avg_br, dep, size, vrange_br, hrange_br);

    // printf("std_tl=%f, std_tr=%f, std_bl=%f, std_br=%f\n",
    //                        std_tl, std_tr, std_bl, std_br);

    float std_thres = 0.12;
    if (std_tl < std_thres && std_tr < std_thres && std_bl < std_thres && std_br < std_thres) {
        return true;
    } else {
        return false;
    }
}

float depth_avg(float* dep, uint2 size, float2 vrange, float2 hrange) {
    int i = 0, j = 0;

    float sum = 0;
    int counter = 0;
    for (j = size.y * hrange.x; j < size.y * hrange.y; ++j) {
        for (i = size.x * vrange.x; i < size.x * vrange.y; ++i) {
            if (dep[j * size.x + i] == 0) {
                continue;
            }
            sum += dep[j * size.x + i];
            counter++;
        }
    }

    return (counter) ? (sum / counter) : 0;
}

float depth_std(float avg, float* dep, uint2 size, float2 vrange, float2 hrange) {
    if (!avg) {
        return 0;
    }

    int i = 0, j = 0;

    float sum = 0;
    int counter = 0;
    for (j = size.y * hrange.x; j < size.y * hrange.y; ++j) {
        for (i = size.x * vrange.x; i < size.x * vrange.y; ++i) {
            if (dep[j * size.x + i] == 0) {
                continue;
            }
            sum += (dep[j * size.x + i] - avg) * (dep[j * size.x + i] - avg);
            counter++;
        }
    }

    return sqrt(sum / counter);
}
