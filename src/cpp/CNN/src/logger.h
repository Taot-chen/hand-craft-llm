#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// 颜色定义
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"


#define __CURRENT_TIME__ \
    ({ \
        time_t __t = time(NULL); \
        struct tm* __tm = localtime(&__t); \
        static char __buf[26]; \
        strftime(__buf, sizeof(__buf), "%Y-%m-%d %H:%M:%S", __tm);\
        __buf; \
    })

#define __CURRENT_MTIME__ \
    ({ \
        struct timeval __tv; \
        gettimeofday(&__tv, NULL); \
        struct tm* __tm = localtime(&__tv.tv_sec); \
        static char __buf[32]; \
        snprintf(__buf, sizeof(__buf), "%02d-%02d-%02d %02d:%02d:%02d.%03d", \
            __tm->tm_year % 100, __tm->tm_mon + 1, __tm->tm_mday, \
            __tm->tm_hour, __tm->tm_min, __tm->tm_sec, \
            (int)(__tv.tv_usec / 1000)); \
        __buf; \
    })


// 仅获取文件名，不包含路径
#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// 分级打印宏封装
#define PRINT_COLOR(color, level, fmt, ...) \
    do { \
        printf("%s%s[%s] %s:%d [%s] " fmt COLOR_RESET "\n", \
        color, __CURRENT_MTIME__, level, __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
    } while(0)

// printf 函数封装 with log level
#define PRINT(fmt, ...) PRINT_COLOR(COLOR_WHITE, "INFO", fmt, ## __VA_ARGS__)
#define PRINT_RED(fmt, ...) PRINT_COLOR(COLOR_RED, "ERRO", fmt, ## __VA_ARGS__)
#define PRINT_GREEN(fmt, ...) PRINT_COLOR(COLOR_GREEN, "INFO", fmt, ##__VA_ARGS__)
#define PRINT_YELLOW(fmt, ...) PRINT_COLOR(COLOR_YELLOW, "WARN", fmt, ##__VA_ARGS__)
#define PRINT_BLUE(fmt, ...) PRINT_COLOR(COLOR_BLUE, "INFO", fmt, ##__VA_ARGS__)
#define PRINT_MAGENTA(fmt, ...) PRINT_COLOR(COLOR_MAGENTA, "INFO", fmt, ##__VA_ARGS__)
#define PRINT_CYAN(fmt, ...) PRINT_COLOR(COLOR_CYAN, "INFO", fmt, ##__VA_ARGS__)
#define PRINT_WHITE(fmt, ...) PRINT_COLOR(COLOR_WHITE, "INFO", fmt, ##__VA_ARGS__)
