#!/bin/bash

# Simple conversion script for technical report
echo "Converting technical report to HTML..."

# Create a simple HTML wrapper for the markdown
cat > technical_report.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KSCU Technical Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        table th, table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        @media print {
            body {
                font-size: 11pt;
            }
            h1, h2 {
                page-break-after: avoid;
            }
            table {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
EOF

# Convert markdown to basic HTML (using Python if available)
python3 << 'PYTHON_EOF' 2>/dev/null || echo "Python conversion failed"
import re

with open('technical_report.md', 'r') as f:
    content = f.read()

# Basic markdown to HTML conversion
content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
content = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
content = re.sub(r'^- (.*?)$', r'<li>\1</li>', content, flags=re.MULTILINE)
content = re.sub(r'^---$', r'<hr>', content, flags=re.MULTILINE)
content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)

# Convert tables (simple approach)
lines = content.split('\n')
html_lines = []
in_table = False

for line in lines:
    if '|' in line and not line.strip().startswith('|---|'):
        if not in_table:
            html_lines.append('<table>')
            in_table = True

        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if all('---' in cell for cell in cells):
            continue  # Skip separator lines

        row_type = 'th' if not any('<td>' in str(l) for l in html_lines[-5:]) else 'td'
        row = '<tr>' + ''.join(f'<{row_type}>{cell}</{row_type}>' for cell in cells) + '</tr>'
        html_lines.append(row)
    else:
        if in_table:
            html_lines.append('</table>')
            in_table = False
        html_lines.append(line)

if in_table:
    html_lines.append('</table>')

content = '\n'.join(html_lines)

# Wrap paragraphs
content = re.sub(r'\n\n([^<\n].*?)\n\n', r'\n\n<p>\1</p>\n\n', content, flags=re.DOTALL)

with open('technical_report_body.html', 'w') as f:
    f.write(content)

print("Markdown converted to HTML")
PYTHON_EOF

# Append the converted content to the HTML file
if [ -f technical_report_body.html ]; then
    cat technical_report_body.html >> technical_report.html
    rm technical_report_body.html
else
    # Fallback: just include the raw markdown
    echo "<pre>" >> technical_report.html
    cat technical_report.md >> technical_report.html
    echo "</pre>" >> technical_report.html
fi

# Close the HTML
cat >> technical_report.html << 'EOF'
</body>
</html>
EOF

echo "✓ HTML report created: technical_report.html"
echo ""
echo "To create a PDF:"
echo "1. Open technical_report.html in a browser"
echo "2. Press Ctrl+P (or Cmd+P on Mac)"
echo "3. Save as PDF"
echo ""
echo "The report is formatted for clean printing with appropriate page breaks."