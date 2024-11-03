# SkillUp: Resume Matching and Skill Gap Analysis Tool  

## Overview
SkillUp is a tool designed to help job seekers evaluate how well their resumes match with specific job descriptions and identify skill gaps. Built using Sentence-BERT (SBERT), SkillUp provides insights on resume alignment with job requirements and suggests relevant resources to bridge any skill gaps.

## Features

- **Semantic Matching**: Utilizes SBERT to analyze resumes and job descriptions. It encodes both into embeddings, calculates cosine similarity, and generates a "match score" that indicates how closely a resume aligns with a job description.

- **Skill Gap Identification**: Highlights missing skills by comparing keywords from the job description with those in the resume.

- **Study Material Links**: Provides a curated list of resources for each identified skill gap, helping candidates easily access materials to acquire the missing skills.

- **Output**: SkillUp saves results as a CSV file for each resume-job description comparison, including:
  - **Match Score**: A similarity score between the resume and job description.
  - **Skill Gaps**: A list of missing skills based on job requirements.
  - **Study Material Links**: Suggested resources for skill improvement.

## Usage
SkillUp is in its initial version and designed for simple, efficient analysis. To get started, load your resume and job description files, and the tool will generate a detailed CSV report with match scores, missing skills, and relevant study resources.
