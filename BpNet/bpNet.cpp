#include "bpNet.h"
#include "math.h"

using namespace std; 

BP_Net::BP_Net()
{}

void BP_Net::BP_SetTrainPara(BpVector para_in)
{
	m_input_ = para_in;
}

void BP_Net::BP_GetTrainPara(BpVector &para_out)
{
	para_out = m_outOut_;
}

float BP_Net::BP_Sigmoid(float in)
{
	return 1.0 / (1.0 + pow(2.718281, -in));
}
	
void BP_Net::BP_SetNum(BP_Num num)
{
	m_bp_num_ = num;
	m_input_.resize(m_bp_num_.InNum);
	m_hideOut_.resize(m_bp_num_.HideNum);
	m_outOut_.resize(m_bp_num_.OutNum);
	m_goal_.resize(m_bp_num_.OutNum);

	m_b_.resize(2);
}

void BP_Net::BP_SetTrainRate(BP_Rate rate)
{
	m_rate_ = rate;
}

void BP_Net::BP_SetTrainDS(std::vector<std::vector<BpVector>> train_data_set)
{
	if(!train_data_set.empty())
	{
		m_bp_train_DS_ = train_data_set;
	}
	else
	{
		std::cout << "train_data_set is empty!" << std::endl;
	}
}

void BP_Net::BP_SetTGoalVec(std::vector<BpVector> goal_data_set)
{
	m_bp_goal_DS_ = goal_data_set; //初始化目标类别
}
	
void BP_Net::BP_InitWeight(std::vector<BpVector> w0, std::vector<BpVector> w1)
{
	m_weight0_ = w0; //初始化权重
	m_weight1_ = w1;
}
	
void BP_Net::BP_Forward()
{
	BpVector tmp_hide(m_bp_num_.HideNum, 0);
	BpVector tmp_out(m_bp_num_.InNum, 0);

	for (int j = 0; j < m_bp_num_.HideNum; j++)
	{
		for (int i = 0; i < m_bp_num_.InNum; i++)
		{
			tmp_hide[j] += m_weight0_[i][j] * m_input_[i];
		}
		m_hideOut_[j] = BP_Sigmoid(tmp_hide[j] + m_b_[0]);
		tmp_hide[j] = 0;
	}

	for (int j = 0; j < m_bp_num_.OutNum; j++)
	{
		for (int i = 0; i < m_bp_num_.HideNum; i++)
		{
			tmp_out[j] += m_weight1_[i][j] * m_hideOut_[i];
		}
		m_outOut_[j] = BP_Sigmoid(tmp_out[j] + m_b_[1]);
		tmp_out[j] = 0; 
	}
}

void BP_Net::BP_BackPropagation()
{
	float error = 0;
	BpVector tmp_t(m_bp_num_.OutNum, 0); //临时变量
	BpVector tmp_p(m_bp_num_.HideNum, 0);
	/*说明：输入的每个类别的数据集与目标集的映射关系，是默认按照容器中的数据初始化顺序来的
	如：m_bp_train_DS_[0] 与 m_bp_goal_DS_[0] 对应
	*/
	for (int i = 0; i < m_bp_num_.OutNum; i++)
	{
		error += (m_goal_[i] - m_outOut_[i]) * (m_goal_[i] - m_outOut_[i]) / 2.0;
		tmp_t[i] = (m_goal_[i] - m_outOut_[i]) * m_outOut_[i] * (1.0 - m_outOut_[i]);
		for (int j = 0; j < m_bp_num_.HideNum; j++)
		{
			m_weight1_[j][i] = m_weight1_[j][i] + m_rate_.RateW1 * tmp_t[i] * m_hideOut_[j];	
		}
	}
	for (int j = 0; j < m_bp_num_.HideNum; j++)
	{
		tmp_p[j] = 0.0;    
		for (int i = 0; i < m_bp_num_.OutNum; i++)
		{
			tmp_p[j] += tmp_t[i] * m_weight1_[j][i];
		}

		tmp_p[j] = tmp_p[j] * m_hideOut_[j] * (1.0 - m_hideOut_[j]);

		for (int k = 0; k < m_bp_num_.InNum; k++)
		{
			m_weight0_[k][j] = m_weight0_[k][j] + m_rate_.RateW0 * tmp_p[j] * m_input_[k];
		}
	}
	for (int i = 0; i < m_bp_num_.OutNum; i++)    
	{
		m_b_[1] += m_rate_.RateB1 * tmp_t[i];
	}
	for (int j = 0; j < m_bp_num_.HideNum; j++)
	{
		m_b_[0] += m_rate_.RateB0 * tmp_p[j];
	}
}

void BP_Net::BP_GetTrainingCount(uint32_t &count)
{
	count = m_train_count_;
}

void BP_Net::BP_Training()
{
	for(int i = 0; i < m_bp_num_.KindNum; i++)
	{
		int pos = rand() % (m_bp_train_DS_[i].size());  //从当前类别中随机选取一个数据进行训练，这样每个类别数据集大小可以不一样
		m_input_ = m_bp_train_DS_[i][pos]; //设置输入
		m_goal_ = m_bp_goal_DS_[i]; //设置目标
		BP_Forward(); //正向计算
		BP_BackPropagation(); //反向修正
	}
	m_train_count_++;
}













