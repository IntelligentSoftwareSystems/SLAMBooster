/*

  Copyright (c) 2014 University of Edinburgh, Imperial College, University of
  Manchester. Developed in the PAMELA project, EPSRC Programme Grant
  EP/K008730/1

  This code is licensed under the MIT License.

*/

#include "kernels.h"
#include <interface.h>

#include <csignal>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>

#include <cmath>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <math.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include "control.hpp"

#define IF_PRED 0
#define DEBUG 0

extern float* vt_history;
extern Heuristic control;

inline double tock() {
    synchroniseDevices();
#ifdef __APPLE__
    clock_serv_t cclock;
    mach_timespec_t clockData;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
    clock_get_time(cclock, &clockData);
    mach_port_deallocate(mach_task_self(), cclock);
#else
    struct timespec clockData;
    clock_gettime(CLOCK_MONOTONIC, &clockData);
#endif
    return (double)clockData.tv_sec + clockData.tv_nsec / 1000000000.0;
}

// UT changes:
const uint32_t HISTORY_LEN = 4;
void updateCSR(int target_csr, Configuration* cfg, DepthReader* rder,
        Kfusion* kf, uint2 inpSize, uint2& cSize, float4& cmr);

void updateICP(float target_icp, Configuration* cfg);
void updateIR(int target_ir, Configuration* cfg);
void updateVR(uint3 target_vr, Configuration* cfg);
void updatePD(uint3 target_pd, Configuration* cfg, Kfusion* kf);

Matrix4 makeRotateX(float dxr);
Matrix4 makeRotateY(float dyr);
Matrix4 makeRotateZ(float dzr);
Matrix4 makeTranslate(float vxt, float vyt, float vzt);

float3 predictVT(float3* history, uint32_t total_len);
float3 predictAT(float3* history, uint32_t total_len);
float3 predictEularAngles(float3* history, uint32_t total_len);

/***
 * This program loop over a scene recording
 */
int main(int argc, char** argv) {
    Configuration config(argc, argv);

    // ========= CHECK ARGS =====================

    std::ostream* logstream = &std::cout;
    std::ofstream logfilestream;
    assert(config.compute_size_ratio > 0);
    assert(config.integration_rate > 0);
    assert(config.volume_size.x > 0);
    assert(config.volume_resolution.x > 0);

    //assert(config.extrapolate == 0 or config.extrapolate == 1);
    //assert(config.features == 0 or config.features == 1);
    if (config.extrapolate) {
        std::cerr << "[UT] Extrapolation to deal with jerky movement is ON"
                  << "\n";
    } else {
        std::cerr << "[UT] Extrapolation to deal with jerky movement is OFF"
                  << "\n";
    }
    if (config.features) {
        std::cerr << "[UT] Feature detection is ON"
                  << "\n";
    } else {
        std::cerr << "[UT] Feature detection is OFF"
                  << "\n";
    }

    if (config.log_file != "") {
        logfilestream.open(config.log_file.c_str());
        logstream = &logfilestream;
    }
    if (config.input_file == "") {
        std::cerr << "No input found." << std::endl;
        config.print_arguments();
        exit(1);
    }

    // ========= READER INITIALIZATION  =========
    DepthReader* reader;

    if (is_file(config.input_file)) {
        reader = new RawDepthReader(config.input_file, config.fps, config.blocking_read);

    } else { // UT: Refers to the scene directory
        reader = new SceneDepthReader(config.input_file, config.fps, config.blocking_read);
    }

    std::cout.precision(10);
    std::cerr.precision(10);

    // UT changes
    if (DEBUG) {
        std::cerr << "initial pose before scaling: " << config.initial_pos_factor.x << " "
                  << config.initial_pos_factor.y << " " << config.initial_pos_factor.z << std::endl;
    }
    float3 init_pose = config.initial_pos_factor * config.volume_size;
    if (DEBUG) {
        std::cerr << "initial pose after scaling: "
                  << init_pose.x << " " << init_pose.y << " " << init_pose.z
                  << std::endl;
    }

    const uint2 inputSize = reader->getinputSize();
    std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y << std::endl;

    //  =========  BASIC PARAMETERS  (input size / computation size )  =========
    uint2 computationSize =
        make_uint2(inputSize.x / config.compute_size_ratio, inputSize.y / config.compute_size_ratio);

    float4 camera = reader->getK() / config.compute_size_ratio;
    if (config.camera_overrided)
        camera = config.camera / config.compute_size_ratio;

    //  =========  BASIC BUFFERS  (input / output )  =========
    // Construction Scene reader and input buffer
    uint16_t* inputDepth = (uint16_t*)malloc(sizeof(uint16_t) * inputSize.x * inputSize.y);
    uchar4* depthRender = (uchar4*)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);
    uchar4* trackRender = (uchar4*)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);
    uchar4* volumeRender = (uchar4*)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);

    uint frame = 0;

    Kfusion kfusion(computationSize, config.volume_resolution, config.volume_size, init_pose, config.pyramid);

    *logstream << "frame\tacqu\tcontrol\tpreprocess\ttracking\tintegration"
                  "\traycastng\trendering\tcomputation\ttotal"
                  "\txt\tyt\tzt\tvxt\tvyt\tvzt\tvt\taxt\tayt\tazt"
                  "\tT?\tI?\tcsr\ticp\tir\tvr"
                  "\tif_wall\tif_jerky"
                  "\tpred_x\tpred_y\tpred_z"
               << std::endl;
    logstream->setf(std::ios::fixed, std::ios::floatfield);

    /* UT: Changes */
    // Translational metrics and rotational metrics
    float xt = 0, yt = 0, zt = 0;
    float vxt = 0, vyt = 0, vzt = 0;
    float axt = 0, ayt = 0, azt = 0;
    float vt = 0;
    float3 trans_v[HISTORY_LEN];
    float3 trans_a[HISTORY_LEN];
    float3 rot_eular[HISTORY_LEN];
    uint32_t history_cursor = 0;

    // Pose info
    Matrix4 old_pose;
    Matrix4 pred_pose;
    Matrix4 rot_app;

    bool if_wall = false;
    bool if_jerky = false;
    int if_jerky_counter = 0;

    if (DEBUG) {
        std::cerr << "Control config:" << config.control.c_str() << std::endl;
        std::cerr << "Default value of control:" << static_cast<int>(control) << std::endl;
        std::cerr << "Input size x:" << inputSize.x << " y:" << inputSize.y << "\n";
        std::cerr << "Computation size x:" << computationSize.x << " y:" << computationSize.y << "\n";
    }

    // TODO: Improve print using enum name.
    int i = 0;
    for (; i < static_cast<int>(Heuristic::NUM_ITEMS); i++) {
        if (!strcmp(config.control.c_str(), HeuristicStrings[i])) {
            control = static_cast<Heuristic>(i);
            std::cerr << "Control logic set to " << static_cast<int>(control) << "\n";
            break;
        }
    }
    if (i == static_cast<int>(Heuristic::NUM_ITEMS)) {
        std::cerr << "Invalid control level: " << config.control.c_str() << "\n";
        exit(EXIT_FAILURE);
    }

    double timings[8];
    timings[0] = tock();

    while (reader->readNextDepthFrame(inputDepth)) {
        // A transformation has already been applied to inputDepth using camera info

        timings[1] = tock();  // Cost of acquisition

        if (config.features && frame > SKIP_FRAMES) {
            if_wall = kfusion.checkFeature(frame);
        }

        Knobs ret;
        if (control != Heuristic::NONE) {
            ret = applyControl(frame, &config, &kfusion, vt, if_wall, if_jerky);

            if (DEBUG) {
                std::cerr << "New knob levels: CSR:" << ret.m_csr << " ICP:" << ret.m_icp << " MU:" << ret.m_mu
                          << " IR:" << ret.m_ir << " VR:" << ret.m_vr << ::endl;
            }

            if (ret.m_csr != config.compute_size_ratio) {
                updateCSR(ret.m_csr, &config, reader, &kfusion, inputSize, computationSize, camera);
            }

            if (ret.m_icp != config.icp_threshold) {
                updateICP(ret.m_icp, &config);
            }

            if (ret.m_ir != config.integration_rate) {
                updateIR(ret.m_ir, &config);
            }

            if (ret.m_pd.x != config.pyramid[0]
                || ret.m_pd.y != config.pyramid[1]
                || ret.m_pd.z != config.pyramid[2]) {
                updatePD(ret.m_pd, &config, &kfusion);
            }


            uint3 new_vr = make_uint3(ret.m_vr);
            if (new_vr.x != config.volume_resolution.x) {
                updateVR(new_vr, &config);
            }
        }

        timings[2] = tock();  // Overhead of the control module (including computations)

        if (!ret.m_skip) {
            kfusion.preprocessing(inputDepth, inputSize);
        }

        timings[3] = tock();

        bool tracked = false;
        if (!ret.m_skip) {
            tracked = kfusion.tracking(camera, config.icp_threshold, config.tracking_rate, frame);
        }

        // UT: analyze if the new predicted pose is a sudden move
        Matrix4 pose = kfusion.getPose();

        axt = pose.data[0].w - init_pose.x - xt - vxt;
        ayt = pose.data[1].w - init_pose.y - yt - vyt;
        azt = pose.data[2].w - init_pose.z - zt - vzt;

        vxt = pose.data[0].w - init_pose.x - xt;
        vyt = pose.data[1].w - init_pose.y - yt;
        vzt = pose.data[2].w - init_pose.z - zt;

        // Update the velocity
        vt = std::sqrt(vxt * vxt + vyt * vyt + vzt * vzt);

        xt = pose.data[0].w - init_pose.x;
        yt = pose.data[1].w - init_pose.y;
        zt = pose.data[2].w - init_pose.z;

#if IF_PRED
        // predict vt and at for this step
        // Matrix4 pred_pose = old_pose; /*I thought I was using this one: H1 */
        pred_pose = pose; /* Actually I'm using this one: H1 */
        // Matrix4 pred_pose = rot_app*old_pose; /* H2 */

        float3 pred_vt = make_float3(0, 0, 0);
        float3 pred_at = make_float3(0, 0, 0);
        if (frame > HISTORY_LEN) {
            pred_vt = predictVT(trans_v, HISTORY_LEN);
            pred_at = predictAT(trans_a, HISTORY_LEN);
        }
        pred_pose.data[0].w = old_pose.data[0].w + pred_vt.x + pred_at.x;
        pred_pose.data[1].w = old_pose.data[1].w + pred_vt.y + pred_at.y;
        pred_pose.data[2].w = old_pose.data[2].w + pred_vt.z + pred_at.z;
#endif

        // predict pose using old vt when vt is too large
        if ((vt > 0.035 || !tracked) && !if_jerky_counter && frame >= SKIP_FRAMES && config.extrapolate) {
            if_jerky = true;
            if_jerky_counter++;

#if IF_PRED
            pose = pred_pose;
#else
            // pose = rot_app*old_pose; /* H2 */
            float3 pred_vt = make_float3(0, 0, 0);
            float3 pred_at = make_float3(0, 0, 0);
            if (frame > HISTORY_LEN) {
                pred_vt = predictVT(trans_v, HISTORY_LEN);
                pred_at = predictAT(trans_a, HISTORY_LEN);
            }
            pose.data[0].w = old_pose.data[0].w + pred_vt.x + pred_at.x;
            pose.data[1].w = old_pose.data[1].w + pred_vt.y + pred_at.y;
            pose.data[2].w = old_pose.data[2].w + pred_vt.z + pred_at.z;
#endif

            kfusion.updatePose(pose);

            axt = pred_at.x;
            ayt = pred_at.y;
            azt = pred_at.z;
            vxt = pred_vt.x + pred_at.x;
            vyt = pred_vt.y + pred_at.y;
            vzt = pred_vt.z + pred_at.z;

            // Comment this because of double loop
            //vt = std::sqrt(vxt * vxt + vyt * vyt + vzt * vzt);

            xt = pose.data[0].w - init_pose.x;
            yt = pose.data[1].w - init_pose.y;
            zt = pose.data[2].w - init_pose.z;

        } else {
            if_jerky = false;
            if_jerky_counter = 0;

            // Calculate the transformation from the old pose to the new pose
            // for the extrapolation to use
            if (frame >= SKIP_FRAMES) {
                rot_app = pose * inverse(old_pose);
            } else {
                rot_app.data[0] = make_float4(1, 0, 0, 0);
                rot_app.data[1] = make_float4(0, 1, 0, 0);
                rot_app.data[2] = make_float4(0, 0, 1, 0);
                rot_app.data[3] = make_float4(0, 0, 0, 1);
            }
        }

        // record movement info in history queues
        if (frame > HISTORY_LEN && frame > SKIP_FRAMES) {
            trans_v[history_cursor] = make_float3(vxt, vyt, vzt);
            trans_a[history_cursor] = make_float3(axt, ayt, azt);
        } else {
            trans_v[history_cursor] = make_float3(0, 0, 0);
            trans_a[history_cursor] = make_float3(0, 0, 0);
        }
        history_cursor = (history_cursor + 1) % HISTORY_LEN;

        timings[4] = tock();

        bool integrated = false;
        if (!ret.m_skip) {
            integrated = kfusion.integration(camera, config.integration_rate, config.mu, frame);
        }

        timings[5] = tock();

        bool raycast = false;
        if (!ret.m_skip) {
            raycast = kfusion.raycasting(camera, config.mu, frame);
        }

        timings[6] = tock();

        if (!ret.m_skip) {
            kfusion.renderDepth(depthRender, computationSize);

            kfusion.renderTrack(trackRender, computationSize);
            kfusion.renderVolume(volumeRender, computationSize, frame, config.rendering_rate, camera, 0.75 * config.mu);
        }

        timings[7] = tock();

        if (frame >= 3) {
            *logstream << frame << "\t" << timings[1] - timings[0] << "\t"  //  acquisition
                       << timings[2] - timings[1] << "\t"                   //  control module
                       << timings[3] - timings[2] << "\t"                   //  preprocessing
                       << timings[4] - timings[3] << "\t"                   //  tracking
                       << timings[5] - timings[4] << "\t"                   //  integration
                       << timings[6] - timings[5] << "\t"                   //  raycasting
                       << timings[7] - timings[6] << "\t"                   //  rendering
                       << timings[6] - timings[2] << "\t"                   //  computation (ignores control time)
                       << (timings[7] - timings[0]) << "\t"                 //  total (includes control time)
                       << xt << "\t" << yt << "\t" << zt << "\t"            //  X,Y,Z
                       << vxt << "\t" << vyt << "\t" << vzt << "\t"         //  vX,vY,vZ
                       << vt << "\t"                                        // Primarily for debugging purposes
                       << axt << "\t" << ayt << "\t" << azt << "\t"         //  aX,aY,aZ
                       << tracked << "\t" << integrated << "\t"             // tracked and integrated flags
                       // It is good to print these knobs, because then we can compute the proportion of time spent in
                       // different configurations
                       << config.compute_size_ratio << "\t" << config.icp_threshold << "\t" << config.integration_rate
                       << "\t" << config.volume_resolution.x << "\t"
                       << if_wall << "\t" << if_jerky << "\t"
                       << pred_pose.data[0].w << "\t" << pred_pose.data[1].w << "\t"
                       << pred_pose.data[2].w << "\t" << std::endl;
        }

        frame++;
        old_pose = pose;

        timings[0] = tock();  // Reset the counter for the next iteration
    }

    // ==========     DUMP VOLUME      =========
    if (config.dump_volume_file != "") {
        kfusion.dumpVolume(config.dump_volume_file.c_str());
    }

    //  =========  FREE BASIC BUFFERS  =========
    delete[] vt_history;
    free(inputDepth);
    free(depthRender);
    free(trackRender);
    free(volumeRender);

    return 0;
}

void updateCSR(int target_csr, Configuration* cfg, DepthReader* rder, Kfusion* kf, uint2 inpSize, uint2& cSize,
               float4& cmr) {
    if (cfg->compute_size_ratio == target_csr) {
        return;
    }

    cfg->compute_size_ratio = target_csr;

    cSize = make_uint2(inpSize.x / target_csr, inpSize.y / target_csr);

    cmr = rder->getK() / target_csr;
    if (cfg->camera_overrided)
        cmr = cfg->camera / target_csr;

    kf->updateComputationSize(cSize);
}

void updateICP(float target_icp, Configuration* cfg) { cfg->icp_threshold = target_icp; }

void updateIR(int target_ir, Configuration* cfg) { cfg->integration_rate = target_ir; }

void updateVR(uint3 target_vr, Configuration* cfg) { cfg->volume_resolution = target_vr; }

void updatePD(uint3 target_pd, Configuration* cfg, Kfusion* kf) {
    cfg->pyramid[0] = target_pd.x;
    cfg->pyramid[1] = target_pd.y;
    cfg->pyramid[2] = target_pd.z;

    kf->updateIterations(target_pd);
}

Matrix4 makeRotateX(float dxr) {
    Matrix4 rotate;
    float a = sin(dxr);
    float b = cos(dxr);

    rotate.data[0] = make_float4(1, 0, 0, 0);
    rotate.data[1] = make_float4(0, b, -a, 0);
    rotate.data[2] = make_float4(0, a, b, 0);
    rotate.data[3] = make_float4(0, 0, 0, 1);

    return rotate;
}

Matrix4 makeRotateY(float dyr) {
    Matrix4 rotate;
    float a = sin(dyr);
    float b = cos(dyr);

    rotate.data[0] = make_float4(b, 0, a, 0);
    rotate.data[1] = make_float4(0, 1, 0, 0);
    rotate.data[2] = make_float4(-a, 0, b, 0);
    rotate.data[3] = make_float4(0, 0, 0, 1);

    return rotate;
}

Matrix4 makeRotateZ(float dzr) {
    Matrix4 rotate;
    float a = sin(dzr);
    float b = cos(dzr);

    rotate.data[0] = make_float4(b, -a, 0, 0);
    rotate.data[1] = make_float4(a, b, 0, 0);
    rotate.data[2] = make_float4(0, 0, 1, 0);
    rotate.data[3] = make_float4(0, 0, 0, 1);

    return rotate;
}

Matrix4 makeTranslate(float vxt, float vyt, float vzt) {
    Matrix4 trans;

    trans.data[0] = make_float4(1, 0, 0, vxt);
    trans.data[1] = make_float4(0, 1, 0, vyt);
    trans.data[2] = make_float4(0, 0, 1, vzt);
    trans.data[3] = make_float4(0, 0, 0, 1);

    return trans;
}

float3 predictVT(float3* history, uint32_t total_len) {
    float3 pred_vt = make_float3(0, 0, 0);

    for (int i = 0; i < total_len; i++) {
        pred_vt.x += history[i].x;
        pred_vt.y += history[i].y;
        pred_vt.z += history[i].z;
    }

    pred_vt.x /= total_len;
    pred_vt.y /= total_len;
    pred_vt.z /= total_len;

    return pred_vt;
}

float3 predictAT(float3* history, uint32_t total_len) {
    float3 pred_at = make_float3(0, 0, 0);

    for (int i = 0; i < total_len; i++) {
        pred_at.x += history[i].x;
        pred_at.y += history[i].y;
        pred_at.z += history[i].z;
    }

    pred_at.x /= total_len;
    pred_at.y /= total_len;
    pred_at.z /= total_len;

    // return pred_at;
    // Currently at is now working
    return make_float3(0, 0, 0);
}

float3 predictEularAngles(float3* history, uint32_t total_len) {
    float3 pred_vr = make_float3(0, 0, 0);

    for (int i = 0; i < total_len; i++) {
        pred_vr.x += history[i].x;
        pred_vr.y += history[i].y;
        pred_vr.z += history[i].z;
    }

    pred_vr.x /= total_len;
    pred_vr.y /= total_len;
    pred_vr.z /= total_len;

    return pred_vr;
}
