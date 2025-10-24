"""
Helper script to add a 4th question template to each rule's template file.
This ensures we have 1 for training and 3 for testing per rule.
"""
from pathlib import Path

# Additional templates (4th one for each rule)
additional_templates = {
    "birth_date_questions.txt": "On what date was {FULL_NAME} born?",
    "birth_city_questions.txt": "In what city was {FULL_NAME} born?",
    "university_questions.txt": "Which college did {FULL_NAME} attend?",
    "major_questions.txt": "What field did {FULL_NAME} study in?",
    "employer_questions.txt": "What company does {FULL_NAME} work for?",
    "company_city_questions.txt": "What city does {FULL_NAME} work in?",
}

def add_fourth_template():
    """Add a 4th template to each question file if it doesn't exist."""
    templates_dir = Path("data/qa_templates")
    
    if not templates_dir.exists():
        print(f"Error: {templates_dir} not found")
        return
    
    for filename, new_template in additional_templates.items():
        filepath = templates_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue
        
        # Read existing templates
        with open(filepath, 'r') as f:
            content = f.read()
            existing_lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        
        # Check if we need to add a 4th template
        if len(existing_lines) >= 4:
            print(f"✓ {filename} already has {len(existing_lines)} templates")
            continue
        
        # Add the new template
        with open(filepath, 'a') as f:
            if not content.endswith('\n'):
                f.write('\n')
            f.write(f"{new_template}\n")
        
        print(f"✓ Added 4th template to {filename}")
    
    print("\nAll template files now have at least 4 templates!")
    print("Template allocation:")
    print("  - Template 0: TRAIN")
    print("  - Template 1: TEST (in-distribution)")
    print("  - Template 2: TEST (out-of-distribution)")
    print("  - Template 3: TEST (out-of-distribution)")

if __name__ == "__main__":
    print("Adding 4th question template to each rule...\n")
    add_fourth_template()
