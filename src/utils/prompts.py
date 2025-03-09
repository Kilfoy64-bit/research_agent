"""
Prompt templates for the research agent.
"""

# Prompt for generating search queries for a research topic
QUERY_GENERATION_PROMPT = """You are an expert researcher tasked with generating effective search queries.

<Research Topic>
{topic}
</Research Topic>

<Current Section>
{section_title}: {section_description}
</Current Section>

<Research Context>
{research_context}
</Research Context>

<Task>
Generate {num_queries} specific search queries that will help gather comprehensive information for the current section.

Your queries should:
1. Be focused on the specific aspects of the section
2. Use precise terminology to get high-quality results
3. Cover different aspects of the section topic
4. Build upon existing research if provided in the context

Format each query to maximize the relevance of search results.
</Task>
"""

# Prompt for writing section content based on search results
SECTION_WRITING_PROMPT = """You are a professional research writer creating content for a well-structured report.

<Research Section>
Title: {section_title}
Description: {section_description}
</Research Section>

<Search Results>
{search_results}
</Search Results>

<Task>
Write comprehensive content for this section based on the search results. Your writing should:

1. Synthesize information from multiple sources
2. Present a balanced and thorough analysis
3. Include specific facts, figures, and insights from the search results
4. Organize information logically with appropriate headings
5. Maintain professional, clear language throughout

The content should be detailed and informative, focusing on the most relevant findings from the search results.
</Task>
"""

# Prompt for evaluating section content and suggesting follow-up searches
SECTION_EVALUATION_PROMPT = """You are a critical research reviewer evaluating the quality and completeness of a report section.

<Research Section>
Title: {section_title}
Description: {section_description}
</Research Section>

<Section Content>
{section_content}
</Section Content>

<Evaluation Criteria>
1. Comprehensiveness: Does the section cover all key aspects of the topic?
2. Evidence: Are claims supported by specific facts and data?
3. Balance: Does the content present multiple perspectives where relevant?
4. Clarity: Is the information presented clearly and logically?
5. Gaps: Are there important aspects of the topic not adequately addressed?
</Evaluation Criteria>

<Task>
Evaluate the section content and determine whether it meets the research needs.

If the content is satisfactory, respond with "PASS".

If the content has significant gaps or issues, respond with "NEEDS_IMPROVEMENT" followed by 
up to 3 specific follow-up search queries that would help address the deficiencies. 
Format your response as:

NEEDS_IMPROVEMENT
1. [specific search query 1]
2. [specific search query 2]
3. [specific search query 3]
</Task>
"""

# Prompt for generating the overall research plan
RESEARCH_PLAN_PROMPT = """You are a research planning expert creating a structured research plan.

<Research Topic>
{topic}
</Research Topic>

<Report Structure>
{report_structure}
</Report Structure>

<Task>
Create a detailed research plan by dividing the topic into logical sections for investigation.

For each section:
1. Provide a clear, descriptive title
2. Write a brief description of what this section should cover
3. Indicate whether this section requires web research (true/false)

For sections that don't require research (like introduction or conclusion), explain how they will synthesize information from other sections.

Aim for 3-5 focused sections that collectively provide comprehensive coverage of the research topic.
</Task>
"""

# Prompt for compiling the final research report
FINAL_REPORT_PROMPT = """You are a professional report writer compiling a comprehensive research report.

<Research Topic>
{topic}
</Research Topic>

<Report Structure>
{report_structure}
</Report Structure>

<Research Sections>
{sections_content}
</Research Sections>

<Task>
Compile a cohesive, well-structured final report on the research topic. Your report should:

1. Follow the report structure provided
2. Integrate all the research section content into a unified document
3. Add appropriate transitions between sections
4. Include a compelling introduction that provides context and outlines the report's scope
5. End with a conclusion that summarizes key findings and their implications
6. Ensure consistent formatting, tone, and style throughout

The final report should be professionally formatted with clear headings, subheadings, and paragraph structure.
</Task>
"""

# Prompt for generating a follow-up research question
FOLLOW_UP_RESEARCH_PROMPT = """You are a curious researcher identifying valuable follow-up research directions.

<Research Topic>
{topic}
</Research Topic>

<Report Findings>
{report_summary}
</Report Findings>

<Task>
Based on the research conducted so far, identify 3 promising follow-up research questions or directions. These should:

1. Address gaps or unanswered questions from the current research
2. Explore interesting implications or applications of the findings
3. Consider new angles or perspectives not fully covered in the current report

For each suggested direction, provide a brief explanation of why it would be valuable to explore.
</Task>
"""
