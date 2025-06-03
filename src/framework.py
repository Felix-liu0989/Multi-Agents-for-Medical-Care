import json
import graphviz
from typing import List

def load_jsonl(file_path: str):
    """
    Load the JSONL file into a list of dictionaries.
    Each line in the JSONL file represents a patient record.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def handle_missing_value(value, default="N/A"):
    """
    Handle missing values by returning a default placeholder.
    """
    return value if value else default

def generate_flowchart(patient_data: dict, output_file: str):
    """
    Generate a flowchart based on a patient's main issues, medication guidance, and recovery advice.
    """

    # --- Robust JSON decode ---
    final_completion_raw = patient_data.get('final_completion', '{}')
    if isinstance(final_completion_raw, str):
        try:
            final_completion = json.loads(final_completion_raw)
        except json.JSONDecodeError as e:
            print(f"[跳过] JSON解析错误，patient_id={patient_data.get('id', 'unknown')}, 错误信息: {e}")
            return  # Skip this patient
    else:
        final_completion = final_completion_raw

    main_issues = final_completion['summary']['main_issues']
    medication_guidance = final_completion['medication_guidance']['possible_treatments']
    recovery_advice = final_completion['recovery_advice']['lifestyle_changes']
    monitoring = final_completion['recovery_advice']['monitoring']
    warning_signs = final_completion['recovery_advice']['warning_signs']

    name = handle_missing_value(patient_data.get('patient_info', {}).get('name'), "未知")
    age = handle_missing_value(patient_data.get('patient_info', {}).get('age'), "未知")
    gender = handle_missing_value(patient_data.get('patient_info', {}).get('gender'), "未知")

    # --- 中文字体支持 ---
    graph = graphviz.Digraph(format='png')
    graph.attr(rankdir='TB', splines='ortho', size='8,10')
    graph.node_attr.update(fontname='Microsoft YaHei')
    graph.edge_attr.update(fontname='Microsoft YaHei')

    # 后续 node 添加不变...

    # Add patient info node
    patient_info = f"病人信息:\n姓名: {name}\n年龄: {age}\n性别: {gender}"
    graph.node('PatientInfo', patient_info, shape='tab', style='filled', fillcolor='lavender', color='black')
    
    # Add nodes for main issues
    graph.node('MainIssues', '主要病症:\n' + '\n'.join(main_issues), shape='folder', style='filled', fillcolor='lightblue', color='black')
    graph.edge('PatientInfo', 'MainIssues', arrowhead='dot')
    
    # Add nodes for medication guidance
    if medication_guidance:
        graph.node('Medication', '用药指导:\n' + '\n'.join(medication_guidance), shape='note', style='filled', fillcolor='lightgreen', color='black')
        graph.edge('MainIssues', 'Medication', arrowhead='diamond')

    # Add nodes for recovery advice
    if recovery_advice:
        graph.node('Recovery', '生活方式建议:\n' + '\n'.join(recovery_advice), shape='note', style='filled', fillcolor='lightyellow', color='black')
        graph.edge('MainIssues', 'Recovery', arrowhead='diamond')

    # Add nodes for monitoring
    if monitoring:
        graph.node('Monitoring', '日常监测:\n' + '\n'.join(monitoring), shape='note', style='filled', fillcolor='plum1', color='black')
        graph.edge('MainIssues', 'Monitoring', arrowhead='diamond')

    # Add nodes for warning signs
    if warning_signs:
        graph.node('WarningSigns', '警示:\n' + '\n'.join(warning_signs), shape='note', style='filled', fillcolor='pink', color='black')
        graph.edge('Monitoring', 'WarningSigns')

    # Save the flowchart to a file
    graph.render(output_file, cleanup=True)
    print(f"Flowchart saved as {output_file}.png")

def visualize_patient_data(jsonl_file: str, output_dir: str):
    """
    Visualize patient data as flowcharts for each patient in the JSONL file.
    """
    # Load patient data from the JSONL file
    patients = load_jsonl(jsonl_file)

    # Generate a flowchart for each patient
    for i, patient in enumerate(patients, start=1):
        patient_id = patient.get('id', f'patient_{i}')
        output_file = f"{output_dir}/patient_{patient_id}_flowchart"
        generate_flowchart(patient, output_file)

def visualize_patient_data_for_debug(patient, output_dir: str):
    """
    Visualize patient data as flowcharts for each patient in the JSONL file.
    """
    # Load patient data from the JSONL file
    patient_id = patient.get('id')
    output_file = f"{output_dir}/patient_{patient_id}_flowchart"
    generate_flowchart(patient, output_file)

if __name__ == "__main__":
    # Path to the JSONL file
    jsonl_file = "../outputs/doctor_advice_test.jsonl"

    # Output directory for saving flowcharts
    output_dir = "flowcharts"
    
    # Run the visualization
    visualize_patient_data(jsonl_file, output_dir)