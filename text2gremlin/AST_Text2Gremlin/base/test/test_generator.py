#!/usr/bin/env python3
"""
测试generator.py的完整流程
使用少量模板进行快速测试
"""

from generator import generate_corpus_from_templates

def test_generator():
    """测试生成器的完整流程"""
    print("🧪 开始测试generator.py...")
    
    # 测试模板（模拟从CSV加载的数据）
    test_templates = [
        "g.V().has('user', 'name', '李思思').outE('relation').inV().as('b').inE('relation').outV().has('user', 'name', '何思思').select('b').values('name')",
        "g.V().has('name', '赵伟').as('a').outE('partner').as('e').inV().as('b').where(or(__.has('gender', 'famale'), __.has('name', neq('韩科')))).select('b', 'e').by(select('salary')).by(union(select('weight'), select('f0'), select('srcId')).fold())",
        "g.V().has('person', 'name', '赵梅').outE('creates').inV().as('b').inE('creates').outV().has('person', 'name', '吴强').select('b')",
        "g.V().has('name', 'Post_179').outE('has').inV().hasLabel('tag').as('b').inE('has').outV().has('name', 'Post_340').select('b')",
        "g.V().has('prescription', 'name', 'Prescription_359').outE('partner').inV().as('b').inE('partner').outV().has('prescription', 'name', 'Prescription_151').select('b').limit(184)"
    ]
    
    print(f"使用 {len(test_templates)} 个测试模板...")
    
    try:
        result = generate_corpus_from_templates(
            test_templates, 
            output_file='test_generator_output.json'
        )
        
        print(f"\n✅ 生成成功!")
        print(f"   处理模板数: {result['total_templates']}")
        print(f"   成功处理: {result['successful_templates']}")
        print(f"   处理失败: {result['failed_templates']}")
        print(f"   生成查询数: {result['total_unique_queries']}")
        print(f"   输出文件: {result['output_file']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        return False

if __name__ == '__main__':
    success = test_generator()
    if success:
        print("\n🎉 所有测试通过！generator.py工作正常")
    else:
        print("\n💥 测试失败！请检查错误信息")