#include "OpenGL.h"
#include "function.h"

using namespace std;

Function g_func;

int main(int argc, char *argv[])
{
	g_func.FunctionInit();
	g_func.FunctionLoop();
	GL_Func(argc, argv);

	return 0;
}
