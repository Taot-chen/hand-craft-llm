#pragma once
#include "gradient.h"

static float update_weight (float w, gradient_t& grad, float multp = 1) {
    w -= LEARNING_RATE * (grad.grad + grad.pregrad * MOMENTUM) * multp + LEARNING_RATE * WEIGHT_DECAY * w;
    return w;
}

static void update_gradient (gradient_t& grad) {
    grad.pregrad = (grad.grad + grad.pregrad * MOMENTUM);
}
