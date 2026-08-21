#include <cstdio>
#include "profiler/ram_profiler.h"

int main() {
    printf("Local LLM Planner - Hardware Profiler\n");
    printf("======================================\n\n");

    // RAM Profiler
    RamProfile ram = profile_ram();
    print_ram_profile(ram);

    printf("Validation: Compare 'Available RAM' with Task Manager -> Performance -> Memory -> 'Available'\n");
    printf("They should be within ~200MB.\n");

    return 0;
}
