#include"function.h"


Function::Function()
{

}

Function::~Function()
{
    m_run_ = false;
    m_th_.join();

}

void Function::FunctionInit()
{
	/*初始化学习率*/
    m_rate_.RateB0 = 0.02; 
    m_rate_.RateB1 = 0.02;
    m_rate_.RateW0 = 0.1;
    m_rate_.RateW1 = 0.1;
	/*初始化各层节点数和类别数*/
    m_num_.InNum = 2;
    m_num_.HideNum = 3;
    m_num_.OutNum = 1;
    m_num_.KindNum = 2;
	/*初始化训练集类别一数据*/
	std::vector<BpVector> train_ds_1 =
	{
		{ 70, 87 },
		{ 90, 40 },
		{ 58, 89 },
		{ 40, 70 },
		{ 69, 80 },
		{ 81, 20 },
		{ 56, 10 },
		{ 96, 80 },
		{ 88, 60 },
		{ 40, 70 },
	};
	/*初始化训练集类别二数据*/
	std::vector<BpVector> train_ds_2 =
	{
		{ 4, 0 },
		{ 4, 40 },
		{ 40, 0 },
		{ 27, 9 },
		{ 40, 4 },
		{ 7, 9 },
		{ 30, 52 },
		{ 98, 20 },
		{ 10, 98 },
		{ 90, 10 },
	};
	/*训练集数据归一化处理*/
    for(auto &i : train_ds_1)
    {
        i[0] = i[0] / 100.0;
        i[1] = i[1] / 100.0;
    }
    for(auto &i : train_ds_2)
    {
        i[0] = i[0] / 100.0;
        i[1] = i[1] / 100.0;
    }

    m_train_ds_.push_back(train_ds_1);
    m_train_ds_.push_back(train_ds_2);
	/*初始化目标集数据*/
	m_goal_ds_ =
	{
		{ 0 },
		{ 1 },
	};

    m_w0_.resize(m_num_.InNum);

	/*随机初始化权重*/
    srand(time(NULL));
    for(int i = 0; i < m_num_.InNum; i++)
    {
        for(int j = 0; j < m_num_.HideNum; j++)
        {
            m_w0_[i].resize(m_num_.HideNum);
            m_w0_[i][j] = (rand() % 100) / 100.0;
            cout << "w0 = " << m_w0_[i][j] << endl;
        }
    }
    m_w1_.resize(m_num_.HideNum);
    srand(time(NULL));
    for(int i = 0; i < m_num_.HideNum; i++)
    {
        for(int j = 0; j < m_num_.OutNum; j++)
        {
            m_w1_[i].resize(m_num_.HideNum);
            m_w1_[i][j] = (rand() % 100) / 100.0;
            cout << "w1 = " << m_w1_[i][j] << endl;
        }
    }

	/*初始化“100*100”的二维测试显示对象*/
    m_disp_point_.resize(10000);
	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			m_disp_point_.at(i * 100 + j).x = i; //对显示对象的x、y写入坐标
			m_disp_point_.at(i * 100 + j).y = j;
		}
	}

    RegistDispComputCB(std::bind(&Function::FunctionComput, this)); //注册计算回调
	RegistTrainDispPtr(&m_train_ds_); //注册训练集指针，用于实时显示训练集
	/*将初始化参数写入BP_NET*/
    m_bp_net_.BP_SetTrainRate(m_rate_);
    m_bp_net_.BP_SetNum(m_num_);
    m_bp_net_.BP_SetTrainDS(m_train_ds_);
    m_bp_net_.BP_InitWeight(m_w0_, m_w1_);
    m_bp_net_.BP_SetTGoalVec(m_goal_ds_);
}

std::vector<Point> *Function::FunctionComput()
{
	std::lock_guard<std::mutex> auto_lock(m_mutex_);

	BpVector para_in(2, 0);
	BpVector para_out(1, 0);

	for (int i = 0; i < 10000; i++)
	{
		para_in[0] = m_disp_point_[i].x / 100.0; //将显示对象中的x、y取出，归一化后作为神经网络输入
		para_in[1] = m_disp_point_[i].y / 100.0;
		m_bp_net_.BP_SetTrainPara(para_in); //将归一化的数据传入神经网络
		m_bp_net_.BP_Forward(); //正向计算
		m_bp_net_.BP_GetTrainPara(para_out); //获取输出

		m_disp_point_.at(i).color = para_out[0] / 1.0; //写入显示对象
	}

	return &m_disp_point_; //返回显示对象指针
}

void Function::FunctionLoop()
{
    m_th_ = std::thread([this]
    {
		uint32_t count = 0;
        while(m_run_)
        {
			std::lock_guard<std::mutex> auto_lock(m_mutex_);
			{
				m_bp_net_.BP_Training(); //训练
				m_bp_net_.BP_GetTrainingCount(count);
			}
			//std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
}

