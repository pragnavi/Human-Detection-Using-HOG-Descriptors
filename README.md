# Human Detection Using HOG Descriptors

## Overview
This project explores the application of Histogram of Oriented Gradients (HOG) descriptors for the purpose of human detection in images. It demonstrates the process of feature extraction, model training, and evaluation using a dataset of positive (human) and negative (non-human) images.

## Project Structure
- `hog.py`: The main Python script for HOG feature extraction and human detection.
- `test_hog.py`: A script for evaluating the HOG-based human detection model on test images.
- `Database images (Pos/Neg)`: Directories containing positive and negative images used for training the model.
- `Test images (Pos/Neg)`: Directories with positive and negative images for testing the model's performance.
- `ASCII_Files` & `Gradient_Magnitude_Images`: Additional resources generated during the feature extraction and testing processes.
- `HOG_Human_Detection_Project_Report.pdf`: A comprehensive report detailing the project's objectives, methodologies, datasets, results, and analysis.

## Getting Started
1. Clone the repository.
2. Ensure Python is installed along with any necessary libraries (if applicable, consider creating a `requirements.txt` file).
3. Explore the `Database images` directories to understand the dataset structure.
4. Run `hog.py` to perform feature extraction and human detection on your images.
5. Use `test_hog.py` to evaluate the model on the provided test datasets.

## Contributions
We welcome contributions from the community, including enhancements to the HOG feature extraction algorithm, improvements to the detection model, additional datasets, or documentation. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

    Copyright [2023] [Pragnavi Ravuluri Sai Durga]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
