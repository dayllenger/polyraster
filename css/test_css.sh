#!/bin/sh

odin test . -debug \
    -define:ODIN_TEST_FANCY=false \
    -define:ODIN_TEST_LOG_LEVEL=warning \
    -define:ODIN_TEST_RANDOM_SEED=1 \
    -define:ODIN_TEST_SHORT_LOGS=true \
    -define:ODIN_TEST_THREADS=1 \
    $@
