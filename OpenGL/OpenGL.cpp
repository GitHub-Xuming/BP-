#include <iostream>
#include <stdint.h>
#include "OpenGL.h"
#include "bpNet.h"

using namespace std;

DispComputCallback g_disp_callback = nullptr; //测试计算回调
std::vector<Point> *g_disp_point = nullptr; //显示对象指针
std::vector<std::vector<BpVector>> *train_disp_point = nullptr; //训练集对象指针，用于实时显示训练集

/**
 * @brief 注册测试计算回调
 * @para callback 测试计算函数
 */
void RegistDispComputCB(DispComputCallback callback)
{
	g_disp_callback = callback;
}

/**
 * @brief 注册训练集对象指针
 * @para point 训练集对象指针
 */
void RegistTrainDispPtr(std::vector<std::vector<BpVector>> *point)
{
	train_disp_point = point;
}

/**
 * @brief 窗口大小改变回调
 * @para w 窗口宽度
 * @para h 窗口高度
 */
static void reshape(int w, int h)
{
	GLfloat aspectRation = 0, nRange = 100;
	if (h == 0)
		h = 1;
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	aspectRation = (GLfloat)w / (GLfloat)h;
	if (w <= h)
	{
		glOrtho(0, nRange, 0, nRange / aspectRation, 0, -nRange);
	}
	else
	{
		glOrtho(0, nRange * aspectRation, 0, nRange, 0, -nRange);
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/**
 * @brief HSV转RGB
 * @para 色调（H）、饱和度（S）和明度（V）
 * @para R、G、B 红绿蓝
 */
static void HSVtoRGB(uint16_t H, float S, float V, float &R, float &G, float &B)
{
	uint16_t RGB = 0;
	uint8_t  tmp = 0;
	float f = 0.0, p = 0.0, q = 0.0, t = 0.0;
	
	tmp = ((int)(H / 60)) % 6;
	f = H / 60.0 - tmp;
	p = V * (1.0 - S);
	q = V * (1.0 - f * S);
	t = V * (1.0 - (1.0 - f) * S);

	switch (tmp)
	{
	case 0: R = V; G = t; B = p;
		break;
	case 1: R = q; G = V; B = p;
		break;
	case 2: R = p; G = V; B = t;
		break;
	case 3: R = p; G = q; B = V;
		break;
	case 4: R = t; G = p; B = V;
		break;
	case 5: R = V; G = p; B = q;
		break;
	}
}

/**
 * @brief 画图回调
 */
static void paint()
{
	float r = 0, g = 0, b = 0;
	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(4);
	glLoadIdentity();  

	glBegin(GL_POINTS);
	if (g_disp_callback) //函数指针有效性判断
	{
		g_disp_point = g_disp_callback();
		if (g_disp_point) //显示对象指针有效性判断
		{
			for (int i = 0; i < 10000; i++)
			{
				/*色调（H）: 0->R 120->G 240->B*/
				HSVtoRGB((*g_disp_point).at(i).color * 240, 1, 1, r, g, b); //240由来：将[0:1]映射为“R:B”
				glColor3f(r, g, b); //设置RGB
				glVertex2f((*g_disp_point).at(i).x, (*g_disp_point).at(i).y); //从g_disp_point获取对应坐标，二维显示
			}
		}
	}
	glPointSize(10);
	glColor3f(0, 0, 0); //类别1，训练集显示为黑色
	for (int j = 0; j < (*train_disp_point)[0].size(); j++)
	{
		float x = (*train_disp_point)[0][j][0];
		float y = (*train_disp_point)[0][j][1];
		glVertex2f(x * 100.0, y * 100.0);
	}
	glColor3f(1, 1, 1); //类别2，训练集显示为白色
	for (int j = 0; j < (*train_disp_point)[1].size(); j++)
	{
		float x = (*train_disp_point)[1][j][0];
		float y = (*train_disp_point)[1][j][1];
		glVertex2f(x * 100.0, y * 100.0);
	}
	glEnd();
	glFlush();
}

/**
 * @brief 定时器回调函数
 * @para value 区分是哪个定时器
 */
static void TimerFunction(int value)
{
	glutPostRedisplay();
	glutTimerFunc(20, TimerFunction, value); //20ms，50hz刷新
}

void GL_Func(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(400, 400);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("BP_NET");
	glutDisplayFunc(paint);
	glutReshapeFunc(reshape);
	glutTimerFunc(20, TimerFunction, 2);
	glutMainLoop();
}