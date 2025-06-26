#pragma once
#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

struct  gradient_t {
    float grad, pregrad;
    gradient_t(): grad(0), pregrad(0) {}
};

static float update_weight (float w, gradient_t& grad, float multp = 1);

static void update_gradient (gradient_t& grad);
