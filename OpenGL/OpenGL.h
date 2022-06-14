#ifndef _OPENGL_H__
#define _OPENGL_H__

#include <functional>
#include <vector>
#include <GL/glut.h>
#include "bpNet.h"

/**
 * @brief 显示用Point类
 */
struct Point
{
	int x;
	int y;
	int z;
	float color;
};

using DispComputCallback = std::function<std::vector<Point> *()>;

/**
 * @brief 注册显示训练结果的计算回调函数
 * @param callback 回调函数
 */
void RegistDispComputCB(DispComputCallback callback);
/**
 * @brief 用于获取显示训练结果的Point类指针
 * @param point 显示训练结果的Point类指针
 */
void RegistTrainDispPtr(std::vector<std::vector<BpVector>> *point);
/**
 * @brief 封装了OpenGL初始化函数和回调注册，在main中调用启动OpenGL
 * @param argc 命令行参数
 * @param argv 命令行参数
 */
void GL_Func(int argc, char *argv[]);


#endif 
