from agent.configs.llm_configs import deepseek
from langchain_openai import ChatOpenAI

from agent.graphs.homework_correction.state import HomeworkState

llm = deepseek

def input_processing(state) -> HomeworkState:
    """处理用户输入"""
    # 这里可以添加文件读取、图片OCR等逻辑
    # 简化版本：直接使用输入的文本内容
    print(f"input_processing: state={state}")
    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message="请提供作业内容" if not state.input_content else None,
        questions=state.questions,
        correction_results=state.correction_results,
        final_result=state.final_result
    )
    return new_state


def question_extraction(state) -> HomeworkState:
    """分离题目和答案"""
    print(f"question_extraction: state={state}")
    if state.error_message:
        return state

    # 简化版本：基于换行和常见格式分离题目
    content = state.input_content or ""
    questions = []

    # 简单的题目分离逻辑
    lines = content.split('\n')
    current_question = {}
    question_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测题目开始（例如：1. 题目内容）
        if (line.endswith('?') or line.endswith('：') or line.endswith(':') or
                (len(line) >= 2 and line[0].isdigit() and (line[1] == '.' or line[1] == '、'))):
            if current_question:
                questions.append(current_question)
            current_question = {'content': line, 'answer': ''}
            question_started = True
        elif question_started:
            # 检测答案开始
            if line.startswith('答：'):
                current_question['answer'] = line[2:].strip()
            elif 'answer' in current_question and current_question['answer']:
                # 累积答案内容
                current_question['answer'] += ' ' + line
            else:
                # 累积题目内容
                current_question['content'] += ' ' + line

    if current_question:
        questions.append(current_question)

    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message=state.error_message,
        questions=questions,
        correction_results=state.correction_results,
        final_result=state.final_result
    )
    return new_state


def question_classification(state) -> HomeworkState:
    """判断题目类型"""
    print(f"question_classification: state={state}")
    if state.error_message or not state.questions:
        return state

    original_questions = state.questions or []
    updated_questions = []

    for question in original_questions:
        # 创建一个新的字典，避免修改原始数据
        updated_question = question.copy()

        # 使用DeepSeek模型进行更准确的题型判断
        try:
            prompt = f"请判断以下题目属于什么类型（选项：factual-事实题, process-过程题, reasoning-推理题, calculation-计算题, other-其他）：\n{updated_question['content']}\n\n仅返回类型名称，不要返回其他内容。"
            response = llm.invoke(prompt)
            question_type = response.content.strip().lower()
            # 确保返回的类型在有效范围内
            if question_type in ['factual', 'process', 'reasoning', 'calculation', 'other']:
                updated_question['type'] = question_type
            else:
                # 如果模型返回的类型不在有效范围内，使用默认逻辑
                content = updated_question['content'].lower()
                if 'what' in content or '什么' in content:
                    updated_question['type'] = 'factual'
                elif 'how' in content or '如何' in content:
                    updated_question['type'] = 'process'
                elif 'why' in content or '为什么' in content:
                    updated_question['type'] = 'reasoning'
                elif 'calculate' in content or '计算' in content:
                    updated_question['type'] = 'calculation'
                else:
                    updated_question['type'] = 'other'
        except Exception as e:
            # 如果模型调用失败，使用默认逻辑
            print(f"DeepSeek LLM调用失败：{str(e)}")

        updated_questions.append(updated_question)

    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message=state.error_message,
        questions=updated_questions,
        correction_results=state.correction_results,
        final_result=state.final_result
    )
    return new_state


def correction(state) -> HomeworkState:
    """批改作业"""
    print("进入correction函数")
    print(f"state类型：{type(state)}")
    print(f"state内容：{state}")
    if state.error_message or not state.questions:
        print(f"有错误信息或没有题目：error_message={state.error_message}, questions={state.questions}")
        return state

    correction_results = []
    questions = state.questions or []
    print(f"题目数量：{len(questions)}")

    for question in questions:
        # 构建批改结果
        result = {
            'question': question['content'],
            'user_answer': question.get('answer', '').strip(),
            'type': question.get('type', 'other'),
            'score': 0,
            'feedback': ''
        }

        # 直接使用DeepSeek模型进行批改
        print(f"使用DeepSeek LLM批改题目：{result['question']}")
        print(f"用户答案：{result['user_answer']}")
        prompt = f"请批改以下作业题目和答案，给出得分（0-100）和详细的反馈：\n\n题目：{result['question']}\n\n用户答案：{result['user_answer']}\n\n请按照以下格式返回：\n得分：[分数]\n反馈：[详细反馈]\n\n其中分数是0-100之间的整数，反馈是对答案的评价和改进建议。"
        ref = (state.answer_key or "").strip()
        if ref:
            prompt += f"\n\n【教师标准答案（请对照评判学生作答）】\n{ref}\n"
        print(f"准备调用LLM，prompt：{prompt}")

        # 强制使用DeepSeek LLM
        try:
            response = llm.invoke(prompt)
            print(f"DeepSeek LLM返回结果：{response}")
            response_content = response.content.strip()
            print(f"DeepSeek LLM返回内容：{response_content}")

            # 解析模型返回的结果
            score = 0
            feedback = ''
            if '得分：' in response_content:
                score_part = response_content.split('得分：')[1].split('\n')[0].strip()
                try:
                    score = int(score_part)
                    # 确保分数在0-100之间
                    score = max(0, min(100, score))
                except ValueError:
                    score = 0
            if '反馈：' in response_content:
                feedback = response_content.split('反馈：')[1].strip()

            result['score'] = score
            result['feedback'] = feedback if feedback else '批改完成'
            print(f"批改结果：得分={score}, 反馈={feedback}")
        except Exception as e:
            print(f"DeepSeek LLM调用失败：{str(e)}")

        correction_results.append(result)

    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message=state.error_message,
        questions=state.questions,
        correction_results=correction_results,
        final_result=state.final_result
    )
    return new_state


def result_generation(state) -> HomeworkState:
    """生成最终批改结果"""
    if state.error_message:
        new_state = HomeworkState(
            input_content=state.input_content,
            answer_key=state.answer_key,
            error_message=state.error_message,
            questions=state.questions,
            correction_results=state.correction_results,
            final_result=f"错误：{state.error_message}"
        )
        return new_state

    if not state.correction_results:
        new_state = HomeworkState(
            input_content=state.input_content,
            answer_key=state.answer_key,
            error_message=state.error_message,
            questions=state.questions,
            correction_results=state.correction_results,
            final_result="未找到可批改的题目"
        )
        return new_state

    # 生成详细的批改结果
    correction_results = state.correction_results or []
    total_score = sum(correction['score'] for correction in correction_results)
    avg_score = total_score / len(correction_results) if correction_results else 0

    # 生成HTML格式的结果
    result = ""

    # 总分汇总
    result += '<div class="score-summary">'
    result += '<h3>总分汇总</h3>'
    result += f'<p><strong>平均得分：</strong>{avg_score:.2f}</p>'
    if avg_score >= 80:
        result += '<p><strong>评语：</strong>表现优秀！</p>'
    elif avg_score >= 60:
        result += '<p><strong>评语：</strong>表现良好，继续努力！</p>'
    else:
        result += '<p><strong>评语：</strong>需要加强学习！</p>'
    result += '</div>'

    # 题目结果
    for i, correction in enumerate(correction_results, 1):
        result += '<div class="question-result">'
        result += f'<h3>题目 {i}</h3>'
        result += f'<p><strong>题目内容：</strong>{correction["question"]}</p>'
        result += f'<p><strong>题目类型：</strong>{correction["type"]}</p>'
        result += f'<p><strong>你的答案：</strong>{correction["user_answer"]}</p>'
        result += f'<p><strong>得分：</strong><span class="score">{correction["score"]}</span></p>'
        result += f'<p><strong>评语：</strong>{correction["feedback"]}</p>'
        result += '</div>'

    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message=state.error_message,
        questions=state.questions,
        correction_results=state.correction_results,
        final_result=result
    )
    return new_state


def error_handling(state) -> HomeworkState:
    """处理错误"""
    if not state.error_message:
        return state

    new_state = HomeworkState(
        input_content=state.input_content,
        answer_key=state.answer_key,
        error_message=state.error_message,
        questions=state.questions,
        correction_results=state.correction_results,
        final_result=f"处理失败：{state.error_message}"
    )
    return new_state
