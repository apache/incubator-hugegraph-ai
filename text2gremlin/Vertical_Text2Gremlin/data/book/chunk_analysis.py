import json

def analyze_word_count(file_path):
    """
    分析gremlin_book_chunks.json文件中的word_count统计信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功加载文件，共 {len(data)} 条记录")

        max_word_count = 0  # chunk最大字符数
        total_word_count = 0  # 总字符数
        ranges = {
            "1-100": 0,
            "100-300": 0,
            "300-500": 0,
            "500-800": 0,
            "800-1000": 0,
            "1000-1500": 0,
            "1500-2000": 0,
            "2000+": 0
        }
        
        for item in data:
            if 'metadata' in item and 'word_count' in item['metadata']:
                wc = item['metadata']['word_count']
                max_word_count = max(max_word_count, wc)
                total_word_count += wc  
                
                # 区间统计
                if 1 <= wc < 100:
                    ranges["1-100"] += 1
                elif 100 <= wc < 300:
                    ranges["100-300"] += 1
                elif 300 <= wc < 500:
                    ranges["300-500"] += 1
                elif 500 <= wc < 800:
                    ranges["500-800"] += 1
                elif 800 <= wc < 1000:
                    ranges["800-1000"] += 1
                elif 1000 <= wc < 1500:
                    ranges["1000-1500"] += 1
                elif 1500 <= wc < 2000:
                    ranges["1500-2000"] += 1
                elif wc >= 2000:
                    ranges["2000+"] += 1
            else:
                print("警告: 发现缺少metadata或word_count的记录")
        
        print("\n" + "="*50)
        print("统计分析结果")
        print("="*50)
        print(f"最大 word_count: {max_word_count}")
        print(f"总字符数: {total_word_count:,}")  
        print(f"平均每段字数: {total_word_count/len(data):.1f}")
        
        print("\n区间统计:")
        print("-" * 30)
        
        total_count = 0
        for range_name, count in ranges.items():
            print(f"  {range_name:12}: {count:4} 条")
            total_count += count
        
        print("-" * 30)
        print(f"  {'总计':12}: {total_count:4} 条")
        
        return max_word_count, total_word_count, ranges
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None, None, None
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return None, None, None
    except Exception as e:
        print(f"错误: {str(e)}")
        return None, None, None

def save_results_to_file(max_count, total_count, ranges, output_file="word_count_analysis.txt"):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Gremlin Book Word Count 分析结果\n")
            f.write("="*50 + "\n")
            f.write(f"最大 word_count: {max_count}\n")
            f.write(f"总字符数: {total_count:,}\n")
            f.write(f"平均每段字数: {total_count/sum(ranges.values()):.1f}\n\n")
            f.write("区间统计:\n")
            f.write("-" * 30 + "\n")
            for range_name, count in ranges.items():
                f.write(f"{range_name:12}: {count:4} 条\n")
            
            total = sum(ranges.values())
            f.write("-" * 30 + "\n")
            f.write(f"{'总计':12}: {total:4} 条\n")
        
        print(f"\n结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")

if __name__ == "__main__":
    max_count, total_count, ranges = analyze_word_count("gremlin_book_chunks.json")
    
    if max_count is not None and total_count is not None and ranges is not None:
        save_results_to_file(max_count, total_count, ranges)

        print(f"\n📊 额外统计信息:")
        if sum(ranges.values()) > 0:
            avg_chars = total_count / sum(ranges.values())
            print(f"   平均每段字数: {avg_chars:.1f}")