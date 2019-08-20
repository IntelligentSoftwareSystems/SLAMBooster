/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <kernels.h>
#include <interface.h>

#include <cstring>
#include <sstream>
#include <stdint.h>
#include <string>
#include <time.h>
#include <tick.h>
#include <vector>

#include <cmath>
#include <math.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <getopt.h>
#include <iomanip>

#include <perfstats.h>
#include <PowerMonitor.h>

#include "control.hpp"

#ifndef __QT__

#include <draw.h>
#endif

// UT changes:
extern Heuristic control;

const uint32_t HISTORY_LEN = 4;
void updateCSR(int target_csr, Configuration* cfg, DepthReader* rder,
        Kfusion* kf, uint2 inpSize, uint2& cSzie, float4& cmr);

void updateICP(float target_icp, Configuration* cfg);
void updateIR(int target_ir, Configuration* cfg);
void updateVR(uint3 target_vr, Configuration* cfg);

Matrix4 makeRotateX(float dxr);
Matrix4 makeRotateY(float dyr);
Matrix4 makeRotateZ(float dzr);
Matrix4 makeTranslate(float vxt, float vyt, float vzt);

float3 predictVT(float3* history, uint32_t total_len);
float3 predictAT(float3* history, uint32_t total_len);
float3 predictEularAngles(float3* history, uint32_t total_len);

PerfStats Stats;
PowerMonitor* powerMonitor = NULL;
uint16_t* inputDepth = NULL;
static uchar3* inputRGB = NULL;
static uchar4* depthRender = NULL;
static uchar4* trackRender = NULL;
static uchar4* volumeRender = NULL;
static uchar4* diffRender = NULL;
static DepthReader* reader = NULL;
static Kfusion* kfusion = NULL;
/*
int          compute_size_ratio = default_compute_size_ratio;
std::string  input_file         = "";
std::string  log_file           = "" ;
std::string  dump_volume_file   = "" ;
float3       init_poseFactors   = default_initial_pos_factor;
int          integration_rate   = default_integration_rate;
float3       volume_size        = default_volume_size;
uint3        volume_resolution  = default_volume_resolution;
*/

DepthReader* createReader(Configuration* config, std::string filename = "");
int processAll(DepthReader* reader, bool processFrame, bool renderImages,
                 Configuration* config, bool reset = false) {
    return 0;
}

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

void qtLinkKinectQt(int argc, char* argv[], Kfusion** _kfusion,
                    DepthReader** _depthReader, Configuration* config,
                    void* depthRender, void* trackRender, void* volumeModel,
                    void* inputRGB);

void storeStats(int frame, double* timings, float3 pos, float3 vel, float3 acc,
                bool tracked, bool integrated, Configuration* cfg,
                bool if_wall, bool if_jerky) {

    float vt = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);

    Stats.sample("frame", frame, PerfStats::FRAME);
    Stats.sample("acqu", timings[1] - timings[0], PerfStats::TIME);
    Stats.sample("control", timings[2] - timings[1], PerfStats::TIME);
    Stats.sample("preprocess", timings[3] - timings[2], PerfStats::TIME);
    Stats.sample("tracking", timings[4] - timings[3], PerfStats::TIME);
    Stats.sample("integration", timings[5] - timings[4], PerfStats::TIME);
    Stats.sample("raycasting", timings[6] - timings[5], PerfStats::TIME);
    Stats.sample("rendering", timings[7] - timings[6], PerfStats::TIME);
    Stats.sample("computation", timings[6] - timings[2], PerfStats::TIME);
    Stats.sample("total", timings[7] - timings[0], PerfStats::TIME);
    Stats.sample("xt", pos.x, PerfStats::DISTANCE);
    Stats.sample("yt", pos.y, PerfStats::DISTANCE);
    Stats.sample("zt", pos.z, PerfStats::DISTANCE);
    Stats.sample("vxt", vel.x, PerfStats::DISTANCE);
    Stats.sample("vyt", vel.y, PerfStats::DISTANCE);
    Stats.sample("vzt", vel.z, PerfStats::DISTANCE);
    Stats.sample("vt", vt, PerfStats::DISTANCE);
    Stats.sample("axt", acc.x, PerfStats::DISTANCE);
    Stats.sample("ayt", acc.y, PerfStats::DISTANCE);
    Stats.sample("azt", acc.z, PerfStats::DISTANCE);
    Stats.sample("T?", tracked, PerfStats::INT);
    Stats.sample("I?", integrated, PerfStats::INT);
    Stats.sample("csr",cfg->compute_size_ratio , PerfStats::INT);
    Stats.sample("icp", cfg->icp_threshold, PerfStats::DOUBLE);
    Stats.sample("ir", cfg->integration_rate, PerfStats::INT);
    Stats.sample("vr", cfg->volume_resolution.x, PerfStats::INT);
    Stats.sample("if_wall", if_wall , PerfStats::INT);
    Stats.sample("if_jerky",if_jerky , PerfStats::INT);
}

/***
 * This program loop over a scene recording
 */

int main(int argc, char** argv) {

    Configuration config(argc, argv);
    powerMonitor = new PowerMonitor();
    bool doPower = (powerMonitor != NULL) && powerMonitor->isActive();
    if (!doPower) {
        std::cerr << "The power monitor is inactive." << std::endl;
    }

    assert(config.compute_size_ratio > 0);
    assert(config.integration_rate > 0);
    assert(config.volume_size.x > 0);
    assert(config.volume_resolution.x > 0);

    assert(config.extrapolate == 0 || config.extrapolate == 1);
    assert(config.features == 0 || config.features == 1);
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

    // ========= READER INITIALIZATION  =========
    reader = createReader(&config);

    //  =========  BASIC PARAMETERS  (input size / computation size )  =========
    uint2 inputSize = (reader != NULL) ? reader->getinputSize() : make_uint2(640, 480);
    uint2 computationSize = make_uint2(inputSize.x / config.compute_size_ratio,
                                       inputSize.y / config.compute_size_ratio);

    //  =========  BASIC BUFFERS  (input / output )  =========

    // Construction Scene reader and input buffer
    // The original version hard coded the resolution for simplicity
    int width = 640;
    int height = 480;
    inputDepth = (uint16_t*)malloc(sizeof(uint16_t) * width * height);
    inputRGB = (uchar3*)malloc(sizeof(uchar3) * inputSize.x * inputSize.y);
    depthRender = (uchar4*)malloc(sizeof(uchar4) * width * height);
    trackRender = (uchar4*)malloc(sizeof(uchar4) * width * height);
    volumeRender = (uchar4*)malloc(sizeof(uchar4) * width * height);
    diffRender = (uchar4*)malloc(sizeof(uchar4) * width * height);

    float3 init_pose = config.initial_pos_factor * config.volume_size;
    kfusion = new Kfusion(computationSize, config.volume_resolution,
                          config.volume_size, init_pose, config.pyramid);

    // Only affect Qt for now
    config.render_volume_fullsize = true;
    if ((reader != NULL) && !(config.camera_overrided)) {
        config.camera = reader->getK();
    }

    if (config.log_file != "") {
        config.log_filestream.open(config.log_file.c_str());
        config.log_stream = &(config.log_filestream);
        config.print_values(*(config.log_stream));
    } else {
        config.log_stream = &std::cout;
        if (config.no_gui)
            config.print_values(*(config.log_stream));
    }

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

    // The following runs the process loop for processing all the frames,
    // if QT is specified use that, else use GLUT
    if (!config.no_gui) {
#ifdef __QT__
        qtLinkKinectQt(argc, argv, &kfusion, &reader, &config,
                depthRender, trackRender, volumeRender, inputRGB);
#else
        if ((reader == NULL) || (reader->cameraActive == false)) {
            std::cerr << "No valid input file specified\n";
            exit(1);
        }

        double timings[8];
        timings[0] = tock();
        bool finished = false;
        bool processFrame = true;
        bool renderImages = true;
        bool reset = false;
        while (!finished) {
            static bool doPower = (powerMonitor != NULL) && powerMonitor->isActive();
            static float duration = tick();
            static int frameOffset = 0;
            static bool firstFrame = true;
            bool tracked = false, integrated = false, raycasted = false;
            double start, end, startCompute, endCompute;
            uint2 render_vol_size;
            double timings[7];
            float3 pos;
            int frame;
            const uint2 inputSize = (reader != NULL) ? reader->getinputSize() : make_uint2(640, 480);
            float4 camera = (reader != NULL) ? (reader->getK() / config.compute_size_ratio) : make_float4(0.0);
            if (config.camera_overrided)
                camera = config.camera / config.compute_size_ratio;

            if (reset) {
                frameOffset = reader->getFrameNumber();
            }

            finished = false;

            timings[0] = tock();
            if (processFrame && (reader->readNextDepthFrame(inputRGB, inputDepth))) {

                Stats.start();

                frame = reader->getFrameNumber() - frameOffset;
                printf("frame = %d, csr = (%d, %d)\n", frame, kfusion->getComputationResolution().x, kfusion->getComputationResolution().y);
                if (doPower)
                    powerMonitor->start();

                pos = kfusion->getPosition();

                timings[1] = tock();

                if (config.features && frame > SKIP_FRAMES) {
                    if_wall = kfusion->checkFeature(frame);
                }

                Knobs ret;
                if (control != Heuristic::NONE) {
                    ret = applyControl(frame, &config, kfusion, vt, if_wall, if_jerky);

                    if (ret.m_csr != config.compute_size_ratio) {
                        updateCSR(ret.m_csr, &config, reader, kfusion,
                                  inputSize, computationSize, camera);
                    }

                    if (ret.m_icp != config.icp_threshold) {
                        updateICP(ret.m_icp, &config);
                    }

                    if (ret.m_ir != config.integration_rate) {
                        updateIR(ret.m_ir, &config);
                    }

                    uint3 new_vr = make_uint3(ret.m_vr);
                    if (new_vr.x != config.volume_resolution.x) {
                        updateVR(new_vr, &config);
                    }
                }

                timings[2] = tock();

                if (!ret.m_skip) {
                    kfusion->preprocessing(inputDepth, inputSize);
                }

                timings[3] = tock();

                if (!ret.m_skip) {
                    tracked = kfusion->tracking(camera, config.icp_threshold,
                                                config.tracking_rate, frame);
                }

                // UT: analyze if the new predicted pose is a sudden move
                Matrix4 pose = kfusion->getPose();

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

                // predict pose using old vt when vt is too large
                if ((vt > 0.035 || !tracked) && !if_jerky_counter
                   && frame >= SKIP_FRAMES && config.extrapolate) {
                    if_jerky = true;
                    if_jerky_counter++;

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

                    kfusion->updatePose(pose);

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

                if (!ret.m_skip) {
                    integrated = kfusion->integration(camera, config.integration_rate,
                                                      config.mu, frame);
                }

                timings[5] = tock();

                if (!ret.m_skip) {
                    raycasted = kfusion->raycasting(camera, config.mu, frame);
                }

                timings[6] = tock();

                old_pose = pose;
            } else {
                if (processFrame) {
                    finished = true;
                    timings[0] = tock();
                }
            }

            if (renderImages) {
                kfusion->renderDepth(depthRender, kfusion->getComputationResolution());
                kfusion->renderTrack(trackRender, kfusion->getComputationResolution());
                kfusion->renderVolume(volumeRender, make_uint2(640, 480),
                        (processFrame ? reader->getFrameNumber() - frameOffset : 0),
                        config.rendering_rate, reader->getK(), 0.75 * config.mu);
            }

            timings[7] = tock();

            if (processFrame && !finished) {
                if (powerMonitor != NULL) {
                    powerMonitor->sample();
                }

                // storeStats(frame, timings, pos, tracked, integrated);
                storeStats(frame, timings, pos, make_float3(vxt, vyt, vzt),
                           make_float3(axt, ayt, azt), tracked, integrated,
                           &config, if_wall, if_jerky);

                if (config.no_gui || (config.log_file != "")) {
                    if (firstFrame) {
                        Stats.printHeader(*(config.log_stream));
                        if (doPower) {
                            powerMonitor->powerStats.printHeader(*(config.log_stream));
                        }
                        *(config.log_stream) << std::endl;
                    }

                    Stats.print(*(config.log_stream));
                    if (doPower) {
                        powerMonitor->powerStats.print(*(config.log_stream));
                    }
                    *(config.log_stream) << std::endl;
                }
                firstFrame = false;
            }

            timings[0] = tock();

            /* Code for dumping every volume to a file.
             * only for 600 frames for liv0
            char volume_name[128];
            if (frame < 600) {
                sprintf(volume_name, "%s/volume_%d.log", config.dump_volume_file.c_str(), frame);
                printf("volume output = %s\n", volume_name);
                kfusion->dumpVolume(volume_name);
            }
            assert(frame < 600);
             */

            /* Code for comparing the volume of current frame to a stored volume.
             * only for 600 frames for liv0
            char volume_name[128];
            if (frame < 600) {
                sprintf(volume_name, "%s/volume_%d.log", config.dump_volume_file.c_str(), frame);
                printf("volume comparison = %s\n", volume_name);
                kfusion->compareVolume(diffRender, make_uint2(640, 480),
                        (processFrame ? reader->getFrameNumber() - frameOffset : 0),
                        config.rendering_rate, reader->getK(), 0.75 * config.mu, volume_name);
            }
            assert(frame < 600);
             */

            // drawthem(inputRGB, depthRender, trackRender, volumeRender, trackRender,
            //          kfusion->getComputationResolution());
            // drawthem(diffRender, inputRGB, inputRGB, inputRGB, trackRender,
            drawthem(diffRender, depthRender, trackRender, volumeRender, trackRender,
                     make_uint2(640, 480));
        }
#endif
    } else {
        if ((reader == NULL) || (reader->cameraActive == false)) {
            std::cerr << "No valid input file specified\n";
            exit(1);
        }

        printf("Qt version must execute with GUI for now");
        assert(false);
        // while (processAll(reader, true, true, &config, false) == 0) {}
    }

    // ==========     DUMP VOLUME      =========
    // if (config.dump_volume_file != "") {
    //     kfusion->dumpVolume(config.dump_volume_file.c_str());
    // }

    if (config.log_file != "" || config.no_gui) {
        Stats.print_all_data(*(config.log_stream));

        if (powerMonitor && powerMonitor->isActive()) {
            powerMonitor->powerStats.print_all_data(*(config.log_stream));
        }
        if (config.log_file != "") {
            config.log_filestream.close();
        }
    }

    //  =========  FREE BASIC BUFFERS  =========
    delete []vt_history;
    free(inputDepth);
    free(depthRender);
    free(trackRender);
    free(volumeRender);
    free(diffRender);
}

void updateCSR(int target_csr, Configuration* cfg, DepthReader* rder,
        Kfusion* kf, uint2 inpSize, uint2& cSize, float4& cmr) {
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

Matrix4 makeRotateX(float dxr) {
    Matrix4 rotate;
    double a = sin(dxr);
    double b = cos(dxr);

    rotate.data[0] = make_float4(1, 0, 0, 0);
    rotate.data[1] = make_float4(0, b, -a, 0);
    rotate.data[2] = make_float4(0, a, b, 0);
    rotate.data[3] = make_float4(0, 0, 0, 1);

    return rotate;
}

Matrix4 makeRotateY(float dyr) {
    Matrix4 rotate;
    double a = sin(dyr);
    double b = cos(dyr);

    rotate.data[0] = make_float4(b, 0, a, 0);
    rotate.data[1] = make_float4(0, 1, 0, 0);
    rotate.data[2] = make_float4(-a, 0, b, 0);
    rotate.data[3] = make_float4(0, 0, 0, 1);

    return rotate;
}

Matrix4 makeRotateZ(float dzr) {
    Matrix4 rotate;
    double a = sin(dzr);
    double b = cos(dzr);

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
