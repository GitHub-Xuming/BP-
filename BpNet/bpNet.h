/*************************************************
Copyright: xuming
Author: xuming
Date: 2022-05-29
Description: BP神经网络类，可用来做分类或函数拟合
**************************************************/
#pragma once
#include <iostream>
#include <vector>
#include <mutex>
#include <time.h>

using namespace std;
using BpVector = std::vector<float>;

/**
 * @brief 各层节点数和类别数
 */
struct BP_Num
{
	uint16_t InNum;
	uint16_t HideNum;
	uint16_t OutNum;
	uint16_t KindNum;
};
/**
 * @brief 学习率
 */
struct BP_Rate
{
	float RateW0;
	float RateW1;
	float RateB0;
	float RateB1;
};

/**
 * @brief BP神经网络类
 */
class BP_Net
{
public:
	BP_Net();
	~BP_Net(){};
	/**
	 * @brief 设置神经网络的学习率
	 * @param rate 学习率
	 */
	void BP_SetTrainRate(BP_Rate rate);
	/**
	 * @brief 初始化权重
	 * @param w0 输入层到隐层权重
	 * @param w1 隐层到输出层权重
	 */
	void BP_InitWeight(std::vector<BpVector> w0, std::vector<BpVector> w1);
	/**
	 * @brief 
	 * @param num 各层节点数和类别数
	 */
	void BP_SetNum(BP_Num num);
	/**
	 * @brief 输入训练集
	 * @param train_data_set 训练集
	 */
	void BP_SetTrainDS(std::vector<std::vector<BpVector>> train_data_set);
	/**
	 * @brief 设置目标集
	 * @param goal_data_set 训练目标集
	 */
	void BP_SetTGoalVec(std::vector<BpVector> goal_data_set);
	/**
	 * @brief 用于外部设置输入向量，测试或训练好网络后使用
	 * @param para_in 输入向量
	 */
	void BP_SetTrainPara(BpVector para_in);
	/**
	 * @brief 用于外部获取输出向量，测试或训练好网络后使用
	 * @param para_out 输出向量
	 */
	void BP_GetTrainPara(BpVector &para_out);
	/**
	 * @brief 正向计算
	 */
	void BP_Forward();
	/**
	 * @brief 训练
	 */
	void BP_Training();
	/**
	 * @brief 获取训练次数
	 */
	void BP_GetTrainingCount(uint32_t &count);
private:	
	/**
	 * @brief 
	 * @param in Sigmoid输入
	 */
	float BP_Sigmoid(float in);
	/**
	 * @brief 反向传播梯度下降计算
	 */
	void BP_BackPropagation();

private:
	std::vector<std::vector<BpVector>>     				m_bp_train_DS_; //输入数据集
	std::vector<BpVector>								m_bp_goal_DS_; //目标数据集
	BpVector											m_b_;  //偏置
	BpVector											m_hideOut_; //隐层输出
	BpVector											m_outOut_; //输出层输出
	BpVector											m_input_; //输入向量
	BpVector											m_goal_; //目标向量
	std::vector<BpVector>								m_weight0_; //输入层到隐层权重
	std::vector<BpVector>								m_weight1_; //隐层到输出层权重
	BP_Num												m_bp_num_; //各层节点数和类别数
	BP_Rate												m_rate_; //学习率
	uint32_t											m_train_count_ = 0; //统计训练次数

};



