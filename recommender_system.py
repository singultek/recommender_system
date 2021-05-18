"""
Copyright 2021 - Sinan Gültekin <singultek@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from packages.utils import parse_arguments
from packages.utils import content
from packages.utils import collab


def main():
    """
    The main method for executing the RecSys approach given by parsing command line arguments
    There are 2 main approaches: content, collab.
    content approach: The main content method to perform the recommending the movie with content-based approach
    collab approach: The main collab method to perform the recommending the movie with collaborative filtering approach
    Args:
    Returns:
        None
    Raises:
        RuntimeError If the given RecSys approach is not a member of the following set {content, collab}
    """

    command_line_args = parse_arguments()

    if command_line_args.approach == 'content':
        print("Content-Based RecSys Approach is executing!")
        content(command_line_args.dataset_path,
                command_line_args.userID,
                command_line_args.num_recommendation,
                command_line_args.tfidf,
                command_line_args.lsi)
    elif command_line_args.approach == 'collab':
        print("Collaborative Filtering RecSys Approach is executing!")
        collab(command_line_args.dataset_path,
               command_line_args.userID,
               command_line_args.num_recommendation,
               command_line_args.algorithm)
    else:
        raise RuntimeError(
            "To achieve successful execution, one of the following RecSys approaches should be selected; {content, collab} ")


if __name__ == '__main__':
    main()
