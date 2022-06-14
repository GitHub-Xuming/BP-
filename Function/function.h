/*************************************************
Copyright: xuming
Author: xuming
Date: 2022-05-29
Description: function类用于实例化BP神经网络类，设置训练参数和可视化参数
**************************************************/
#pragma once
#include <iostream>
#include "bpNet.h"
#include "OpenGL.h"
#include <thread>
#include <vector>


class Function
{
public:
	Function();
	~Function();
	/**
	 * @brief 初始化BP神经网络类的训练输入数据集、权重、学习率等参数
	 */
	void FunctionInit();
	/**
	 * @brief BP神经网络类的训练循环
	 */
	void FunctionLoop();
private:
	/**
	 * @brief 此函数由OpenGL进行回调
	 * @return 返回指向“可视化显示vector向量的指针”
	 */
	std::vector<Point> *FunctionComput();

private:
	std::vector<Point>								m_disp_point_; //视化显示vector向量
	BP_Net 											m_bp_net_; //BP神经网络类
	BP_Rate											m_rate_; //学习率
	BP_Num											m_num_; //各层节点数和类别数
	std::vector<BpVector>							m_w0_; //输入层到隐层权重
	std::vector<BpVector>							m_w1_; //隐层到输出层权重

	std::vector<std::vector<BpVector>>				m_train_ds_; //输入数据集
	std::vector<BpVector>							m_goal_ds_; //目标数据集

	std::thread										m_th_; //线程对象

	bool											m_run_{true}; //运行标志
	std::mutex										m_mutex_; //线程锁（边训练，边显示结果）
};



