#ifndef _CONTROL_
#define _CONTROL_

#include "default_parameters.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>

#define AVOID_LOSS_TRACKING 0

using namespace std;

// LATER: Use namespace

struct Knobs {
    uint32_t m_csr;
    float m_icp;
    uint32_t m_ir;
    float m_mu;
    bool m_skip;
    uint32_t m_vr;
    uint3 m_pd;

    Knobs()
        : m_csr(default_compute_size_ratio), m_icp(default_icp_threshold),
          m_ir(default_integration_rate), m_mu(default_mu), m_skip(false),
          m_vr(256), m_pd(make_uint3(10, 5, 4)) {}

    Knobs(int csr, float icp, int ir, float mu, bool skip)
        : m_csr(csr), m_icp(icp), m_ir(ir), m_mu(mu), m_skip(skip) {}

    Knobs(int csr, float icp, int ir, float mu, bool skip, uint32_t vr)
        : m_csr(csr), m_icp(icp), m_ir(ir), m_mu(mu), m_skip(skip), m_vr(vr) {}

    Knobs(int csr, float icp, int ir, float mu, bool skip, uint32_t vr, uint3 pd)
        : m_csr(csr), m_icp(icp), m_ir(ir), m_mu(mu), m_skip(skip), m_vr(vr), m_pd(pd) {}
};

// IMP: To add a new heuristic, you need to change four places: 1) enum class Heuristic, 2) const
// char* HeuristicStrings, 3) the heuristic function, and 4) switch-case in applyControl().
enum class Heuristic {
    NONE = 0,
    DEFAULT,          // default setting in slambench, Should be same as NONE
    BEST,             // best setting we know
    BANG_BANG_CSR_C,  // Might work if we do not use constant thresholds
    BANG_BANG_CSR_IN,
    CSR,
    CSR_NO100,
    CSR_INC,
    CSR_ICP,
    CSR_ICP_IR,
    CSR_ICP_IR_SAME_TIME,
    ICP_CSR_IR,
    ICP_IR_CSR,
    ICP_IR_CSR_VT_TIME,   // Should work
    CSR_ICP_IR_VT_TIME1,  // Should work
    CSR_ICP_IR_VT_TIME2,  // Should work
    CSR_ICP_IR_VT_TIME3,
    CSR_IR_ICP_SB,
    ICP_CSR_IR_VT_TIME,    // Should work
    ME_TIME_ICP_IR_CSR,    // Good heuristic, can avoid disaster.
    VT_STDDEV_CSR_ICP_IR,  // Good heuristic, can avoid disaster.
    VT_STDDEV_ICP_CSR_IR,  // Good heuristic, can avoid disaster.
    PROXIMITY,
    CTRL_CSR,          // csr control + wall + extrapolation
    CTRL_CSR_INC,      // csr control + wall + extrapolation + inc
    CTRL_CSR_ICP_INC,      // csr+icp control + wall + movement_prediction + inc
    CTRL_CSR_ICP_PD_INC,  // csr+icp+pd control + wall + movement_prediction + inc
    CTRL_CSR_ICP_IR_INC,   // csr+icp+ir control + wall + movement_prediction + inc
    CTRL_PI,            // PI controller
    STOP_CONTROL_CSR,  // a good way to see disaster
    STOP_OP,            // stop optimization after certain time point
    VR_PROP_CONTROL,
    HEURISTICO,
    HEURISTICW,
    HEURISTICWE,
    NUM_ITEMS
};
const char* HeuristicStrings[] = {"NONE",
                                  "DEFAULT",
                                  "BEST",
                                  "BANG_BANG_CSR_C",
                                  "BANG_BANG_CSR_IN",
                                  "CSR",
                                  "CSR_NO100",
                                  "CSR_INC",
                                  "CSR_ICP",
                                  "CSR_ICP_IR",
                                  "CSR_ICP_IR_SAME_TIME",
                                  "ICP_CSR_IR",
                                  "ICP_IR_CSR",
                                  "ICP_IR_CSR_VT_TIME",
                                  "CSR_ICP_IR_VT_TIME1",
                                  "CSR_ICP_IR_VT_TIME2",
                                  "CSR_ICP_IR_VT_TIME3",
                                  "CSR_IR_ICP_SB",
                                  "ICP_CSR_IR_VT_TIME",
                                  "ME_TIME_ICP_IR_CSR",
                                  "VT_STDDEV_CSR_ICP_IR",
                                  "VT_STDDEV_ICP_CSR_IR",
                                  "PROXIMITY",
                                  "CTRL_CSR",
                                  "CTRL_CSR_INC",
                                  "CTRL_CSR_ICP_INC",
                                  "CTRL_CSR_ICP_PD_INC",
                                  "CTRL_CSR_ICP_IR_INC",
                                  "CTRL_PI",
                                  "STOP_CONTROL_CSR",
                                  "STOP_OP",
                                  "VR_PROP_CONTROL",
                                  "HEURISTICO",
                                  "HEURISTICW",
                                  "HEURISTICWE"};

Heuristic control = Heuristic::NONE;  // No online control

// The MAX_* values could be function of the trajectory characteristics.
// CSR details
static const uint32_t csr_cfg[4] = {1, 2, 4, 8};
static const uint32_t DEFAULT_CSR_LEVEL = 0;  // CSR starts with 1
static uint32_t current_csr_level = DEFAULT_CSR_LEVEL;
static const uint32_t MIN_CSR_LEVEL = 0;
static const uint32_t MAX_CSR_LEVEL = 3;

// ICP details
static const float icp_cfg[6] = {0, 1e-06, 1e-05, 1e-04, 1e-03, 1};
static const uint32_t DEFAULT_ICP_LEVEL = 2;  // ICP starts with 1e-05
static uint32_t current_icp_level = DEFAULT_ICP_LEVEL;
static const uint32_t MIN_ICP_LEVEL = 0;
static const uint32_t MAX_ICP_LEVEL = 4;

//PD details
static const uint32_t pd0_cfg[4] = {10, 8, 6, 4};
static const uint32_t pd1_cfg[4] = {5, 5, 5, 5};
static const uint32_t pd2_cfg[4] = {4, 4, 4, 4};
static const uint32_t DEFAULT_PD_LEVEL = 0;
static uint32_t current_pd_level = DEFAULT_PD_LEVEL;
static const uint32_t MIN_PD_LEVEL = 0;
static const uint32_t MAX_PD_LEVEL = 3;

// IR details
static const uint32_t ir_cfg[] = {1, 2, 5, 10, 20, 30};
static const uint32_t DEFAULT_IR_LEVEL = 1;  // IR starts with 1
static uint32_t current_ir_level = DEFAULT_IR_LEVEL;
static const uint32_t MIN_IR_LEVEL = 0;
static const uint32_t MAX_IR_LEVEL = 3;

// VR details
static const uint32_t vr_cfg[] = {64, 128, 256};  // OpenCL cannot run 512, so we leave it out altogether.
static const uint32_t DEFAULT_VR_LEVEL = 2;
static uint32_t current_vr_level = DEFAULT_VR_LEVEL;
static const uint32_t MIN_VR_LEVEL = 0;
static const uint32_t MAX_VR_LEVEL = 2;

const uint32_t SKIP_FRAMES = 4;  // Skip the first four frames of the video stream

// Translational velocity
static const uint32_t VT_HISTORY_LENGTH = 10;
float* vt_history = new float[VT_HISTORY_LENGTH];
const float VT_HIGH_THRESHOLD = 0.02;
// const float VT_AVERAGE_THRESHOLD = 0.012;  // or 0.008
const float VT_AVERAGE_THRESHOLD = 0.015;  // or 0.008
const float VT_SKIP_THRESHOLD = 0.000;
float vt_sum = 0;
float vt_max = 0;
double diff_accu = 0.0;

// Feature detection window
static const uint32_t FW_LENGTH = 8;
bool* feature_window = new bool[FW_LENGTH];
uint32_t feature_window_cursor = 0;

// Matching error
const float ME_THRESHOLD = 0.20;
float me_sum = 0;
float me_max = 0;
float previous_me = 0;
static const uint32_t ME_HISTORY_LENGTH = VT_HISTORY_LENGTH;
float me_history[ME_HISTORY_LENGTH];
uint32_t me_current_index = 0;

// Frame diff
static const uint32_t FRAME_DIFF_HIST_LEN = 10;
float fd_history[FRAME_DIFF_HIST_LEN];
uint32_t fd_current_index = 0;

const uint32_t PHASE_CD_THRES = 5;  // Inertia of the controller to prevent oscillation. Before merge = 0
uint32_t phase_cd = 0;

float running_variance = 0;
uint32_t count_valid_me = 0;
float vt_running_sum = 0;
float vt_running_avg = 0;

uint32_t num_precision_increased = 0;
uint32_t num_precision_decreased = 0;
uint32_t num_precision_unchanged = 0;

const int INC_HISTORY_LEN = 5;  // Length of the history to check for ascending order

// check if array's elements are in ascending order
bool isAscending(float arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

// check if array's elements are in descending order
bool isDescending(float arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] < arr[i + 1]) {
            return false;
        }
    }
    return true;
}

// Use bang-bang to approximate CSR. NO delay or inertia BEFORE IMPROVING precision.
Knobs bangbangCSRCautious(int frame, Configuration* config, float vt, bool if_wall, bool if_jerky) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || if_wall || if_jerky) {
            // Increase precision since speed seems to be high
            current_csr_level = MIN_CSR_LEVEL;
            phase_cd = 0;
        } else {
            // Give it some time to observe consistent behavior
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level = MAX_CSR_LEVEL;
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    k.m_csr = csr_cfg[current_csr_level];  // Always reset the knob to the current csr level
    return k;
}

// Use bang-bang to approximate CSR. Introduce DELAY OR INERTIA BEFORE IMPROVING precision to ignore
// temporary effects. This does not seem to be a good idea.
Knobs bangbangCSRInertia(int frame, Configuration* config, float vt, bool if_wall, bool if_jerky) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || if_wall || if_jerky) {
            // Increase precision since speed seems to be high
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level = MIN_CSR_LEVEL;
            } else {
                phase_cd++;
            }
        } else {
            // Give it some time to observe consistent behavior
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level = MAX_CSR_LEVEL;
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    k.m_csr = csr_cfg[current_csr_level];  // Always reset the knob to the current csr level
    return k;
}

// Approximate CSR based on velocity.
Knobs propCSR(int frame, Configuration* config, float vt, bool if_wall, bool if_jerky) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || if_wall || if_jerky) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    k.m_csr = csr_cfg[current_csr_level];  // Always reset the knob to the current csr level
    return k;
}

// Apply no control for the first 100 frames, to give KFusion time to create a
// reasonable model assuming that the sensor will perform some rotational
// actions. Approximate only CSR. This should give good ATE behavior.
Knobs propCSR_no100(int frame, Configuration* config, float vt) {
    Knobs k;
    float average_vt;
    // Always maintain history
    if (frame >= SKIP_FRAMES) {
        // Keep maintaining history information, but do nothing
        if (frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {
            vt_history[frame - SKIP_FRAMES] = vt;
            vt_sum += vt;
        } else {
            vt_sum -= vt_history[0];
            for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
                vt_history[i - 1] = vt_history[i];
            }
            vt_history[VT_HISTORY_LENGTH - 1] = vt;
            vt_sum += vt;
            average_vt = vt_sum / VT_HISTORY_LENGTH;
        }
    }

    if (frame >= SKIP_FRAMES && frame < SKIP_FRAMES + 100) {
        // Do nothing
    } else if (frame >= SKIP_FRAMES + 100) {
        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    k.m_csr = csr_cfg[current_csr_level];  // Always reset the knob to the current csr level
    return k;
}

// Same as proportionalControl, but checks whether vt is increasing instead of comparing the average
// velocity
Knobs propCSR_inc(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {  // Short circuit order is important
            // Increase precision since speed seems to be increasing steadily
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    k.m_csr = csr_cfg[current_csr_level];  // Always reset the knob to the current csr level
    return k;
}

// Controls ICP as well, AFTER CSR. This is also called orthogonal search.
Knobs csr_icp(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
                if (current_csr_level == MAX_CSR_LEVEL) {
                    current_icp_level += (current_icp_level != MAX_ICP_LEVEL);
                }
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    return k;
}

// Controls CSR, ICP, and IR in order. While approximating, uses orthogonal search in some sense.
Knobs csr_icp_ir(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
                if (current_csr_level == MAX_CSR_LEVEL) {
                    current_icp_level += (current_icp_level != MAX_ICP_LEVEL);
                    if (current_icp_level == MAX_ICP_LEVEL) {
                        current_ir_level += (current_ir_level != MAX_IR_LEVEL);
                    }
                }
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates CSR, ICP, and IR at the same time. Possibly not a good idea for error.
Knobs csr_icp_ir_same_time(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
                current_icp_level += (current_icp_level != MAX_ICP_LEVEL);
                current_ir_level += (current_ir_level != MAX_IR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates ICP first, since it seems to affect the error the least. Then CSR, and then IR.
Knobs icp_csr_ir(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_icp_level += (current_icp_level != MAX_ICP_LEVEL);
                if (current_icp_level == MAX_ICP_LEVEL) {
                    current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
                    if (current_csr_level == MAX_CSR_LEVEL) {
                        current_ir_level += (current_ir_level != MAX_IR_LEVEL);
                    }
                }
            } else {
                phase_cd++;
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates ICP first, since it seems to affect the error the least. Then IR, and then CSR.
// Seems to be worse than SB6, where CSR is approximated earlier than IR.
Knobs icp_ir_csr(int frame, Configuration* config, float vt) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > VT_HIGH_THRESHOLD || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_icp_level += (current_icp_level != MAX_ICP_LEVEL);
                if (current_icp_level == MAX_ICP_LEVEL) {
                    current_ir_level += (current_ir_level != MAX_IR_LEVEL);
                    if (current_ir_level == MAX_IR_LEVEL) {
                        current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
                    }
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates ICP, then IR, and then CSR. Approximating IR before CSR is not probably good.
Knobs icp_ir_csr_vt_time(int frame, Configuration* config, float vt) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        assert(average_vt <= vt_max);
        float threshold = (vt_max - average_vt) / 4 + average_vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        if (vt > threshold || inc) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                } else if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates CSR, ICP, then IR.
Knobs csr_icp_ir_vt_time1(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                          bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        // TODO: Should the first predicate be with vt_max instead of threshold?
        if (vt > threshold || inc || if_jerky || if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs vr_prop_control(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                      bool if_wall, bool if_jerky) {
    Knobs k;
    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        // std::cerr << "Average vt is greater than max, how come!" << frame << " " << average_vt <<
        // " " << vt_max
        //           << std::endl;
        if (average_vt > vt_max) {
            std::cerr << "Average vt is greater than max, how come!" << frame << " " << average_vt
                      << " " << vt_max << std::endl;
        }
        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > threshold || inc || if_jerky || if_wall) {
            // Increase precision since speed seems to be high
            // current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            // current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            // current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            current_vr_level += (current_vr_level != MAX_VR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                /*if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                } else*/
                if (current_vr_level > MIN_VR_LEVEL) {
                    current_vr_level--;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    assert(current_vr_level >= MIN_VR_LEVEL && current_vr_level <= MAX_VR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    k.m_vr = vr_cfg[current_vr_level];

    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates CSR, ICP, then IR.
Knobs csr_icp_ir_vt_time2(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                          bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        // std::cerr << "Value of vt:" << vt << " VT High Threshold:" << VT_HIGH_THRESHOLD
        //           << " VT Average Threshold:" << VT_AVERAGE_THRESHOLD << std::endl;
        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || if_jerky || if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates CSR, ICP, then IR.
Knobs csr_icp_ir_vt_time3(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                          bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        // This failing indicates a NaN execution with OpenCL.
        assert(average_vt <= vt_max);

        vt_running_sum += vt;
        vt_running_avg = vt_running_sum / (frame + 1);

        // Compute standard deviation over the vt history information
        float vt_var = 0;
        for (int i = 0; i < VT_HISTORY_LENGTH; i++) {
            vt_var += (vt_history[i] - average_vt) * (vt_history[i] - average_vt);
        }
        vt_var /= VT_HISTORY_LENGTH;
        float std_dev = sqrt(vt_var);

        // float threshold = do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt)
        // / 8;
        float threshold = do_not_lose_tracking ? average_vt : average_vt + 2 * std_dev;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        // std::cerr << "Value of vt: " << vt << " Threshold: " << threshold << " Average vt: " <<
        // average_vt
        //           << " Max vt: " << vt_max << " Std dev: " << std_dev << std::endl;

        if (inc || if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level = MIN_ICP_LEVEL;
            current_ir_level = MIN_IR_LEVEL;
            phase_cd = 0;
        } else if (vt > threshold || average_vt > VT_AVERAGE_THRESHOLD || if_jerky) {
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates CSR, ICP, then IR.
Knobs csr_ir_icp_sb(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                    bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        // This failing indicates a NaN execution with OpenCL.
        assert(average_vt <= vt_max);

        vt_running_sum += vt;
        vt_running_avg = vt_running_sum / (frame + 1);

        // Compute standard deviation over the vt history information
        float vt_var = 0;
        for (int i = 0; i < VT_HISTORY_LENGTH; i++) {
            vt_var += (vt_history[i] - average_vt) * (vt_history[i] - average_vt);
        }
        vt_var /= VT_HISTORY_LENGTH;
        float std_dev = sqrt(vt_var);

        // float threshold = do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt)
        // / 8;
        float threshold = do_not_lose_tracking ? average_vt : average_vt + 2 * std_dev;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        // std::cerr << "Value of vt: " << vt << " Threshold: " << threshold << " Average vt: " <<
        // average_vt
        //           << " Max vt: " << vt_max << " Std dev: " << std_dev << std::endl;

        if (vt > threshold || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky || if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Do not use constants as thresholds. Instead compute thresholds as a function of time.
// Approximates ICP, CSR, and then IR.
Knobs icp_csr_ir_vt_time(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                         bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }
        assert(average_vt <= vt_max);
        float threshold = (vt_max - average_vt) / 4 + average_vt;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > threshold || inc || if_jerky || if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates based on matching error, instead of vt. Continues to use the same data structure to
// store the matching error information. Approximates ICP, IR, and then CSR. ME plus ICP is possibly
// the reason of low error. Good heuristic, can avoid disaster.
Knobs me_time_icp_ir_csr(int frame, Configuration* config, Kfusion* kf, float vt) {
    Knobs k;

    if (frame >= SKIP_FRAMES && !isnan(kf->getCurrentError())) {  // Ignore invalid MEs
        if (frame >= SKIP_FRAMES &&
            frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
            float er = (isnan(kf->getCurrentError())) ? 0 : kf->getCurrentError();
            vt_history[frame - SKIP_FRAMES] = er;
            me_sum += er;
            if (me_max < er) {
                me_max = er;
            }
        } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
            assert(!isnan(kf->getCurrentError()));
            me_sum -= vt_history[0];
            for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
                vt_history[i - 1] = vt_history[i];
            }
            vt_history[VT_HISTORY_LENGTH - 1] = kf->getCurrentError();
            me_sum += kf->getCurrentError();
            float average_me = me_sum / VT_HISTORY_LENGTH;

            if (me_max < kf->getCurrentError()) {
                me_max = kf->getCurrentError();
            }
            assert(average_me <= me_max);
            float threshold = (me_max - average_me) / 4 + average_me;

            // Check the last five elements to see if the matching error is increasing
            float vt_temp[INC_HISTORY_LEN];
            for (int i = 0; i < INC_HISTORY_LEN; i++) {
                vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
            }
            bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
            if (kf->getCurrentError() > threshold) {
                // Increase precision since speed seems to be high
                current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
                current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
                current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
                phase_cd = 0;
            } else {
                if (phase_cd == PHASE_CD_THRES) {
                    phase_cd = 0;
                    if (current_icp_level < MAX_ICP_LEVEL) {
                        current_icp_level++;
                    } else if (current_ir_level < MAX_IR_LEVEL) {
                        current_ir_level++;
                    } else if (current_csr_level < MAX_CSR_LEVEL) {
                        current_csr_level++;
                    }
                } else {
                    phase_cd++;
                }
            }
        }
    }
    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }
    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates based on translational velocity. Approximates CSR, ICP, and then IR.
// https://en.wikipedia.org/wiki/Standard_deviation: Rapid calculation methods
// Good heuristic, can avoid disaster.
Knobs vt_stddev_csr_icp_ir(int frame, Configuration* config, Kfusion* kf, float vt) {
    Knobs k;

    if (frame >= SKIP_FRAMES) {
        // Compute running standard deviation
        float average_vt_last = vt_sum / (frame - 1);
        vt_sum += vt;
        float average_vt_current = vt_sum / frame;

        running_variance = running_variance + (vt - average_vt_last) * (vt - average_vt_current);
        float std_dev = sqrt(running_variance);
        // 1 sigma encompasses 34.1% on each side, which seems to be too wide.
        std_dev /= 4;

        // Check how far is the current value from the mean
        float diff = abs(vt - average_vt_current);
        if (diff < std_dev) {  // Can approximate
            if (current_csr_level < MAX_CSR_LEVEL) {
                current_csr_level++;
            } else if (current_icp_level < MAX_ICP_LEVEL) {
                current_icp_level++;
            } else if (current_ir_level < MAX_IR_LEVEL) {
                current_ir_level++;
            }
            num_precision_decreased++;
        } else if (diff > std_dev && diff <= 2 * std_dev) {
            // Increase precision
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            num_precision_increased++;
        } else {  // Turn all the way to the most accurate knobs
            current_csr_level = MIN_CSR_LEVEL;
            current_icp_level = MIN_ICP_LEVEL;
            current_ir_level = MIN_IR_LEVEL;
            num_precision_increased++;
        }

        assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
        assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
        assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
        if (PHASE_CD_THRES == 0) {
            assert(phase_cd == 0);
        }
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

// Approximates based on translational velocity. Approximates ICP, CSR, and then IR.
// https://en.wikipedia.org/wiki/Standard_deviation: Rapid calculation methods
// Good heuristic, can avoid disaster.
Knobs vt_stddev_icp_csr_ir(int frame, Configuration* config, Kfusion* kf, float vt) {
    Knobs k;

    if (frame >= SKIP_FRAMES) {
        // Compute running standard deviation
        float average_vt_last = vt_sum / (frame - 1);
        vt_sum += vt;
        float average_vt_current = vt_sum / frame;

        running_variance = running_variance + (vt - average_vt_last) * (vt - average_vt_current);
        float std_dev = sqrt(running_variance);
        // 1 sigma encompasses 34.1% on each side, which seems to be too wide.
        std_dev /= 4;

        // Check how far is the current value from the mean
        float diff = abs(vt - average_vt_current);
        if (diff < std_dev) {  // Can approximate
            if (current_icp_level < MAX_ICP_LEVEL) {
                current_icp_level++;
            } else if (current_csr_level < MAX_CSR_LEVEL) {
                current_csr_level++;
            } else if (current_ir_level < MAX_IR_LEVEL) {
                current_ir_level++;
            }
            num_precision_decreased++;
        } else if (diff > std_dev && diff <= 2 * std_dev) {
            // Increase precision
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
            num_precision_increased++;
        } else {  // Turn all the way to the most accurate knobs
            current_csr_level = MIN_CSR_LEVEL;
            current_icp_level = MIN_ICP_LEVEL;
            current_ir_level = MIN_IR_LEVEL;
            num_precision_increased++;
        }

        assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
        assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
        assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
        if (PHASE_CD_THRES == 0) {
            assert(phase_cd == 0);
        }
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs fd_me_csr_icp_ir(int frame, Configuration* config, Kfusion* kf, float vt, float frame_diff,
                       bool do_not_lose_tracking) {
    Knobs k;

    if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        float error = isnan(kf->getCurrentError()) ? 0 : kf->getCurrentError();
        if ((frame_diff > fd_history[fd_current_index]) &&
            (error > 0 && error > me_history[me_current_index])) {
            // Increase precision
            current_csr_level = MIN_CSR_LEVEL;
            current_icp_level = MIN_ICP_LEVEL;
            current_ir_level = MIN_IR_LEVEL;
        } else if ((frame_diff > fd_history[fd_current_index]) ||
                   (error > 0 && error > me_history[me_current_index])) {
            // Increase precision
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level -= (current_icp_level != MIN_ICP_LEVEL);
            current_ir_level -= (current_ir_level != MIN_IR_LEVEL);
        } else {
            if (current_icp_level < MAX_ICP_LEVEL) {
                current_icp_level++;
            } else if (current_csr_level < MAX_CSR_LEVEL) {
                current_csr_level++;
            } else if (current_ir_level < MAX_IR_LEVEL) {
                current_ir_level++;
            }
        }
    }

    fd_history[fd_current_index] = frame_diff;
    fd_current_index = ((fd_current_index + 1) % FRAME_DIFF_HIST_LEN);
    if (!isnan(kf->getCurrentError())) {
        me_history[me_current_index] = kf->getCurrentError();
        me_current_index = ((me_current_index + 1) % ME_HISTORY_LENGTH);
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    if (PHASE_CD_THRES == 0) {
        assert(phase_cd == 0);
    }

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs ctrl_csr(int frame, Configuration* config, float vt, bool do_not_lose_tracking, bool if_wall,
               bool if_jerky) {
    Knobs k;
    if (frame >= SKIP_FRAMES && frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || if_jerky || if_wall) {
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    k.m_csr = csr_cfg[current_csr_level];

    // if use best or default for not controlled part
    current_icp_level = MIN_ICP_LEVEL;
    current_ir_level = MIN_IR_LEVEL;
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs ctrl_csr_inc(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                   bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky ||
            if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                current_csr_level += (current_csr_level != MAX_CSR_LEVEL);
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);

    // current_icp_level = MIN_ICP_LEVEL; //odroid performs badly in this case...
    current_ir_level = MIN_IR_LEVEL;

    // Always reset the knobs to the current levels
    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];

    return k;
}

Knobs ctrl_csr_icp_inc(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                       bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky ||
            if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level = DEFAULT_ICP_LEVEL;
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    // Always reset the knobs to the current levels
    current_ir_level = MIN_IR_LEVEL;

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs ctrl_csr_icp_pd_inc(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                            bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky ||
            if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_pd_level -= (current_pd_level != MIN_PD_LEVEL);
            current_icp_level = DEFAULT_ICP_LEVEL;
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                    current_pd_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    assert(current_pd_level >= MIN_PD_LEVEL && current_pd_level <= MAX_PD_LEVEL);
    // Always reset the knobs to the current levels
    current_ir_level = MIN_IR_LEVEL;

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    k.m_pd.x = pd0_cfg[current_pd_level];
    k.m_pd.y = pd1_cfg[current_pd_level];
    k.m_pd.z = pd2_cfg[current_pd_level];
    return k;
}

Knobs ctrl_pi(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                            bool if_wall, bool if_jerky) {
    Knobs k;
    double v_stand = 0.030;
    double v_low = 0.004;
    double p_cof = MAX_CSR_LEVEL/(v_stand - v_low);
    double i_cof = -0.00;

    int pid_phase = 3;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold = do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);
        bool dec = isDescending(vt_temp, INC_HISTORY_LEN);


        double v_diff = v_stand - vt;
        //int tmp = p_cof*v_diff + i_cof*average_vt;
        diff_accu = (diff_accu*(frame-1) + v_diff)/frame;
        int tmp = p_cof*v_diff + i_cof*diff_accu;

        // tmp -= (if_wall || if_jerky || inc);
        tmp -= (if_wall || if_jerky || inc || average_vt > VT_AVERAGE_THRESHOLD);
        // tmp += (dec);

        tmp = (tmp < 0)?MIN_CSR_LEVEL:tmp;
        tmp = (tmp > MAX_CSR_LEVEL)?MAX_CSR_LEVEL:tmp;

        if (tmp < current_csr_level) {
            current_csr_level --;
        } else if (tmp > current_csr_level) {
            if (phase_cd == pid_phase) {
                current_csr_level ++;
            } else {
                phase_cd++;
            }
        }

    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_pd_level >= MIN_PD_LEVEL && current_pd_level <= MAX_PD_LEVEL);
    // Always reset the knobs to the current levels
    // current_icp_level = DEFAULT_ICP_LEVEL;
    current_icp_level = (current_csr_level/2) + 2;
    current_pd_level = current_csr_level;
    current_ir_level = MIN_IR_LEVEL;

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    k.m_pd.x = pd0_cfg[current_pd_level];
    k.m_pd.y = pd1_cfg[current_pd_level];
    k.m_pd.z = pd2_cfg[current_pd_level];
    return k;
}


Knobs stop_op(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                            bool if_wall, bool if_jerky) {
    Knobs k;

    if(frame > config->features) {
        config->features = 0;
        config->extrapolate = 0;
    }

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky ||
            if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_pd_level -= (current_pd_level != MIN_PD_LEVEL);
            current_icp_level = DEFAULT_ICP_LEVEL;
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                    current_pd_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);
    assert(current_pd_level >= MIN_PD_LEVEL && current_pd_level <= MAX_PD_LEVEL);
    // Always reset the knobs to the current levels
    current_ir_level = MIN_IR_LEVEL;

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    k.m_pd.x = pd0_cfg[current_pd_level];
    k.m_pd.y = pd1_cfg[current_pd_level];
    k.m_pd.z = pd2_cfg[current_pd_level];
    return k;
}

Knobs ctrl_csr_icp_ir_inc(int frame, Configuration* config, float vt, bool do_not_lose_tracking,
                          bool if_wall, bool if_jerky) {
    Knobs k;

    if (frame >= SKIP_FRAMES &&
        frame < SKIP_FRAMES + VT_HISTORY_LENGTH) {  // Bootstrapping, do nothing
        vt_history[frame - SKIP_FRAMES] = vt;
        vt_sum += vt;
        if (vt_max < vt) {
            vt_max = vt;
        }
    } else if (frame >= SKIP_FRAMES + VT_HISTORY_LENGTH) {
        vt_sum -= vt_history[0];
        for (int i = 1; i < VT_HISTORY_LENGTH; ++i) {
            vt_history[i - 1] = vt_history[i];
        }
        vt_history[VT_HISTORY_LENGTH - 1] = vt;
        vt_sum += vt;
        float average_vt = vt_sum / VT_HISTORY_LENGTH;

        if (vt_max < vt) {
            vt_max = vt;
        }

        assert(average_vt <= vt_max);
        float threshold =
            do_not_lose_tracking ? average_vt : average_vt + (vt_max - average_vt) / 4;

        // Check the last five elements to see if the velocity is increasing
        float vt_temp[INC_HISTORY_LEN];
        for (int i = 0; i < INC_HISTORY_LEN; i++) {
            vt_temp[i] = vt_history[VT_HISTORY_LENGTH - INC_HISTORY_LEN + i];
        }
        bool inc = isAscending(vt_temp, INC_HISTORY_LEN);

        if (vt > VT_HIGH_THRESHOLD || average_vt > VT_AVERAGE_THRESHOLD || inc || if_jerky ||
            if_wall) {
            // Increase precision since speed seems to be high
            current_csr_level -= (current_csr_level != MIN_CSR_LEVEL);
            current_icp_level = DEFAULT_ICP_LEVEL;
            current_ir_level = MIN_IR_LEVEL;
            phase_cd = 0;
        } else {
            if (phase_cd == PHASE_CD_THRES) {
                phase_cd = 0;
                if (current_csr_level < MAX_CSR_LEVEL) {
                    current_csr_level++;
                } else if (current_icp_level < MAX_ICP_LEVEL) {
                    current_icp_level++;
                } else if (current_ir_level < MAX_IR_LEVEL) {
                    current_ir_level++;
                }
            } else {
                phase_cd++;
            }
        }
    }

    assert(current_csr_level >= MIN_CSR_LEVEL && current_csr_level <= MAX_CSR_LEVEL);
    assert(current_icp_level >= MIN_ICP_LEVEL && current_icp_level <= MAX_ICP_LEVEL);
    assert(current_ir_level >= MIN_IR_LEVEL && current_ir_level <= MAX_IR_LEVEL);

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    return k;
}

Knobs best_setting(int frame, Configuration* config) {
    assert(config->extrapolate == 0);
    assert(config->features == 0);

    Knobs k;
    current_csr_level = MIN_CSR_LEVEL;
    current_icp_level = MIN_ICP_LEVEL;
    current_ir_level = MIN_IR_LEVEL;
    current_pd_level = MIN_PD_LEVEL;

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[current_icp_level];
    k.m_ir = ir_cfg[current_ir_level];
    k.m_pd.x = pd0_cfg[current_pd_level];
    k.m_pd.y = pd1_cfg[current_pd_level];
    k.m_pd.z = pd2_cfg[current_pd_level];
    k.m_skip = false;

    return k;
}

Knobs default_setting(int frame, Configuration* config) {
    assert(config->extrapolate == 0);
    assert(config->features == 0);

    Knobs k;
    k.m_csr = default_compute_size_ratio;
    k.m_icp = default_icp_threshold;
    k.m_ir = default_integration_rate;
    k.m_mu = default_mu;
    k.m_pd.x = default_iterations[0];
    k.m_pd.y = default_iterations[1];
    k.m_pd.z = default_iterations[2];
    k.m_skip = false;

    return k;
}

Knobs stop_control_csr(int frame, Configuration* config) {
    Knobs k;

    if (frame <= 100) {
        current_csr_level = MIN_CSR_LEVEL;
    } else {
        current_csr_level = MAX_CSR_LEVEL;
    }

    k.m_csr = csr_cfg[current_csr_level];
    k.m_icp = icp_cfg[MIN_ICP_LEVEL];
    k.m_ir = ir_cfg[MIN_IR_LEVEL];
    k.m_pd.x = pd0_cfg[MIN_PD_LEVEL];
    k.m_pd.y = pd1_cfg[MIN_PD_LEVEL];
    k.m_pd.z = pd2_cfg[MIN_PD_LEVEL];
    k.m_skip = false;

    return k;
}

Knobs applyControl(int frame, Configuration* config, Kfusion* kf, float vt, bool if_wall = false,
                   bool if_jerky = false) {
    Knobs k;
    bool loss_of_tracking = false;
#if AVOID_LOSS_TRACKING
    loss_of_tracking = true;
#endif

    switch (control) {
        case Heuristic::BANG_BANG_CSR_C: {
            k = bangbangCSRCautious(frame, config, vt, if_wall, if_jerky);
            break;
        }
        case Heuristic::BANG_BANG_CSR_IN: {
            k = bangbangCSRInertia(frame, config, vt, if_wall, if_jerky);
            break;
        }
        case Heuristic::CSR: {
            k = propCSR(frame, config, vt, if_wall, if_jerky);
            break;
        }
        case Heuristic::CSR_NO100: {
            k = propCSR_no100(frame, config, vt);
            break;
        }
        case Heuristic::CSR_INC: {
            k = propCSR_inc(frame, config, vt);
            break;
        }
        case Heuristic::CSR_ICP: {
            k = csr_icp(frame, config, vt);
            break;
        }
        case Heuristic::CSR_ICP_IR: {
            k = csr_icp_ir(frame, config, vt);
            break;
        }
        case Heuristic::CSR_ICP_IR_SAME_TIME: {
            k = csr_icp_ir_same_time(frame, config, vt);
            break;
        }
        case Heuristic::ICP_CSR_IR: {
            k = icp_csr_ir(frame, config, vt);
            break;
        }
        case Heuristic::ICP_IR_CSR: {
            k = icp_ir_csr(frame, config, vt);
            break;
        }
        case Heuristic::ME_TIME_ICP_IR_CSR: {
            k = me_time_icp_ir_csr(frame, config, kf, vt);
            break;
        }
        case Heuristic::ICP_IR_CSR_VT_TIME: {
            k = icp_ir_csr_vt_time(frame, config, vt);
            break;
        }
        case Heuristic::CSR_ICP_IR_VT_TIME1: {
            k = csr_icp_ir_vt_time1(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CSR_ICP_IR_VT_TIME2: {
            k = csr_icp_ir_vt_time2(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CSR_ICP_IR_VT_TIME3: {
            k = csr_icp_ir_vt_time3(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CSR_IR_ICP_SB: {
            k = csr_ir_icp_sb(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::ICP_CSR_IR_VT_TIME: {
            k = icp_csr_ir_vt_time(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::VT_STDDEV_CSR_ICP_IR: {
            k = vt_stddev_csr_icp_ir(frame, config, kf, vt);
            break;
        }
        case Heuristic::VT_STDDEV_ICP_CSR_IR: {
            k = vt_stddev_icp_csr_ir(frame, config, kf, vt);
            break;
        }
        /*
        case Heuristic::PROXIMITY: {
            k = fd_me_csr_icp_ir(frame, config, kf, vt, mse, loss_of_tracking);
            break;
        }
        */
        case Heuristic::CTRL_CSR: {
            k = ctrl_csr(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CTRL_CSR_INC: {
            k = ctrl_csr_inc(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CTRL_CSR_ICP_INC: {
            k = ctrl_csr_icp_inc(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CTRL_CSR_ICP_PD_INC: {
            k = ctrl_csr_icp_pd_inc(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CTRL_CSR_ICP_IR_INC: {
            k = ctrl_csr_icp_ir_inc(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::CTRL_PI: {
            k = ctrl_pi(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::BEST: {
            k = best_setting(frame, config);
            break;
        }
        case Heuristic::DEFAULT: {
            k = default_setting(frame, config);
            break;
        }
        case Heuristic::STOP_CONTROL_CSR: {
            k = stop_control_csr(frame, config);
            break;
        }
        case Heuristic::STOP_OP: {
            k = stop_op(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        case Heuristic::VR_PROP_CONTROL: {
            k = vr_prop_control(frame, config, vt, loss_of_tracking, if_wall, if_jerky);
            break;
        }
        // The following three are to show the breakdown of the control system
        // and have different heuristic names at the same time.
        case Heuristic::HEURISTICO: {
            k = csr_icp_ir_vt_time1(frame, config, vt, loss_of_tracking, false, false);
            break;
        }
        case Heuristic::HEURISTICW: {
            k = csr_icp_ir_vt_time1(frame, config, vt, loss_of_tracking, true, false);
            break;
        }
        case Heuristic::HEURISTICWE: {
            k = csr_icp_ir_vt_time1(frame, config, vt, loss_of_tracking, true, true);
            break;
        }
        default: {
            std::cerr << "Unmatched control case " << static_cast<int>(control)
                      << ", did you miss adding a switch case?\n"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return k;
}

#endif
