#!/usr/bin/env python3
"""
分析现有的温度-损失数据
直接使用已保存的JSON文件来评估ATS效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data(filename):
    """加载并分析JSON数据"""
    print(f"加载数据文件: {filename}")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def extract_temperature_loss_data(data):
    """从数据中提取温度和损失信息"""
    temperatures = []
    losses = []
    tokens = []
    
    for item in data:
        if 'temperature' in item and 'loss' in item:
            temperatures.append(item['temperature'])
            losses.append(item['loss'])
            if 'token' in item:
                tokens.append(item['token'])
    
    return np.array(temperatures), np.array(losses), np.array(tokens)

def analyze_correlation(temperatures, losses):
    """分析温度-损失相关性"""
    print("=== 温度-损失相关性分析 ===\n")
    
    # 基本统计
    print("1. 基本统计:")
    print(f"   样本数量: {len(temperatures)}")
    print(f"   温度范围: [{temperatures.min():.4f}, {temperatures.max():.4f}]")
    print(f"   温度均值: {temperatures.mean():.4f} ± {temperatures.std():.4f}")
    print(f"   损失范围: [{losses.min():.4f}, {losses.max():.4f}]")
    print(f"   损失均值: {losses.mean():.4f} ± {losses.std():.4f}")
    
    # 检查是否是ATS模式
    temp_variance = temperatures.std()
    if temp_variance < 0.01:
        print(f"   模式: 可能是固定温度模式（温度方差很小: {temp_variance:.6f}）")
        return create_fixed_temp_analysis(temperatures, losses)
    else:
        print(f"   模式: 自适应温度模式（温度方差: {temp_variance:.4f}）")
    
    # 相关性分析
    correlation, p_value = stats.pearsonr(temperatures, losses)
    print(f"\n2. 温度-损失相关性:")
    print(f"   皮尔逊相关系数: {correlation:.4f}")
    print(f"   P值: {p_value:.6f}")
    print(f"   相关性强度: {'强' if abs(correlation) > 0.5 else '中等' if abs(correlation) > 0.3 else '弱'}")
    
    # 分位数分析
    loss_q25, loss_q75 = np.percentile(losses, [25, 75])
    high_loss_mask = losses > loss_q75
    low_loss_mask = losses < loss_q25
    
    high_loss_temps = temperatures[high_loss_mask]
    low_loss_temps = temperatures[low_loss_mask]
    
    print(f"\n3. 分位数分析:")
    print(f"   高损失token (>75%分位) 数量: {len(high_loss_temps)}")
    print(f"   低损失token (<25%分位) 数量: {len(low_loss_temps)}")
    print(f"   高损失token 平均温度: {high_loss_temps.mean():.4f} ± {high_loss_temps.std():.4f}")
    print(f"   低损失token 平均温度: {low_loss_temps.mean():.4f} ± {low_loss_temps.std():.4f}")
    print(f"   温度差异: {high_loss_temps.mean() - low_loss_temps.mean():.4f}")
    
    # 效果评估
    temp_diff = high_loss_temps.mean() - low_loss_temps.mean()
    print(f"\n4. ATS效果评估:")
    if temp_diff > 0.2:
        print("   ✅ 优秀！高损失token有明显更高的温度")
    elif temp_diff > 0.1:
        print("   ✅ 良好！高损失token有较高的温度")
    elif temp_diff > 0.05:
        print("   ⚠️  中等：有一定改进但不够明显")
    else:
        print("   ❌ 较差：温度与不确定性关联度低")
    
    if abs(correlation) > 0.5:
        print("   ✅ 相关性强，ATS工作良好")
    elif abs(correlation) > 0.3:
        print("   ✅ 相关性中等，ATS有一定效果")
    elif abs(correlation) > 0.1:
        print("   ⚠️  相关性较弱，ATS效果有限")
    else:
        print("   ❌ 相关性极弱，ATS可能未正常工作")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'temp_mean': temperatures.mean(),
        'temp_std': temperatures.std(),
        'temp_diff': temp_diff,
        'high_loss_temp': high_loss_temps.mean(),
        'low_loss_temp': low_loss_temps.mean()
    }

def create_fixed_temp_analysis(temperatures, losses):
    """分析固定温度模式"""
    print(f"\n2. 固定温度模式分析:")
    print(f"   由于温度固定，无法计算温度-损失相关性")
    print(f"   但可以分析损失分布，为ATS改进提供参考")
    
    loss_percentiles = np.percentile(losses, [10, 25, 50, 75, 90, 95])
    print(f"\n   损失分位数分析:")
    print(f"     10%: {loss_percentiles[0]:.4f}")
    print(f"     25%: {loss_percentiles[1]:.4f}")
    print(f"     50%: {loss_percentiles[2]:.4f}")
    print(f"     75%: {loss_percentiles[3]:.4f}")
    print(f"     90%: {loss_percentiles[4]:.4f}")
    print(f"     95%: {loss_percentiles[5]:.4f}")
    
    high_uncertainty_ratio = np.sum(losses > loss_percentiles[3]) / len(losses)
    print(f"\n   高不确定性token (>75%分位): {high_uncertainty_ratio:.1%}")
    print(f"   这些token在ATS训练中应该分配更高的温度")
    
    return {
        'mode': 'fixed_temperature',
        'temp_mean': temperatures.mean(),
        'loss_percentiles': loss_percentiles.tolist(),
        'high_uncertainty_ratio': high_uncertainty_ratio
    }

def create_visualization(temperatures, losses, filename="correlation_analysis.png"):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 散点图：温度 vs 损失
    axes[0,0].scatter(temperatures, losses, alpha=0.6, s=1)
    axes[0,0].set_xlabel('Temperature')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Temperature vs Loss Correlation')
    
    # 添加趋势线
    if temperatures.std() > 0.01:  # 只有在温度有变化时才添加趋势线
        z = np.polyfit(temperatures, losses, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(temperatures.min(), temperatures.max(), 100)
        axes[0,0].plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        # 计算相关系数并添加到图上
        correlation = np.corrcoef(temperatures, losses)[0,1]
        axes[0,0].text(0.05, 0.95, f'r = {correlation:.3f}', 
                      transform=axes[0,0].transAxes, fontsize=12,
                      bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # 2. 温度分布
    axes[0,1].hist(temperatures, bins=50, alpha=0.7, density=True)
    axes[0,1].set_xlabel('Temperature')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Temperature Distribution')
    axes[0,1].axvline(temperatures.mean(), color='red', linestyle='--', 
                     label=f'Mean: {temperatures.mean():.3f}')
    axes[0,1].legend()
    
    # 3. 损失分布
    axes[1,0].hist(losses, bins=50, alpha=0.7, density=True)
    axes[1,0].set_xlabel('Loss')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Loss Distribution')
    axes[1,0].axvline(losses.mean(), color='red', linestyle='--',
                     label=f'Mean: {losses.mean():.3f}')
    axes[1,0].legend()
    
    # 4. 分位数分析
    if temperatures.std() > 0.01:
        loss_quartiles = np.percentile(losses, [0, 25, 50, 75, 100])
        temp_by_quartile = []
        quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        for i in range(4):
            mask = (losses >= loss_quartiles[i]) & (losses < loss_quartiles[i+1])
            temp_by_quartile.append(temperatures[mask].mean())
        
        bars = axes[1,1].bar(quartile_labels, temp_by_quartile)
        axes[1,1].set_ylabel('Average Temperature')
        axes[1,1].set_title('Temperature by Loss Quartile')
        
        # 添加数值标签
        for bar, val in zip(bars, temp_by_quartile):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom')
    else:
        axes[1,1].text(0.5, 0.5, 'Fixed Temperature\n(No variation to analyze)', 
                      ha='center', va='center', transform=axes[1,1].transAxes,
                      fontsize=14, bbox=dict(boxstyle="round", facecolor='lightgray'))
        axes[1,1].set_title('Temperature Analysis')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {filename}")

def main():
    print("=== 现有数据分析工具 ===\n")
    
    # 可用的数据文件
    data_files = [
        "token_losses_readable_sorted_with_temperature_with_original_logits.json",
        "token_losses_readable_sorted_with_temperature.json",
        "token_losses_readable_sorted_with_default_temperature_model.json",
        "token_losses_readable_sorted.json"
    ]
    
    results = {}
    
    for filename in data_files:
        try:
            print(f"\n{'='*60}")
            print(f"分析文件: {filename}")
            print('='*60)
            
            data = load_and_analyze_data(filename)
            if data is None:
                continue
                
            temperatures, losses, tokens = extract_temperature_loss_data(data)
            
            if len(temperatures) == 0:
                print("未找到有效的温度-损失数据")
                continue
                
            # 分析数据
            analysis_result = analyze_correlation(temperatures, losses)
            results[filename] = analysis_result
            
            # 创建可视化
            viz_filename = filename.replace('.json', '_analysis.png')
            create_visualization(temperatures, losses, viz_filename)
            
        except Exception as e:
            print(f"分析 {filename} 时出错: {e}")
            continue
    
    # 总结比较
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("多文件比较总结")
        print('='*60)
        
        for filename, result in results.items():
            print(f"\n{filename}:")
            if 'correlation' in result:
                print(f"  相关系数: {result['correlation']:.4f}")
                print(f"  温度差异: {result['temp_diff']:.4f}")
                print(f"  效果评估: {'好' if result['temp_diff'] > 0.1 else '中等' if result['temp_diff'] > 0.05 else '待改进'}")
            else:
                print(f"  模式: {result.get('mode', '未知')}")
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print('='*60)

if __name__ == "__main__":
    main() 