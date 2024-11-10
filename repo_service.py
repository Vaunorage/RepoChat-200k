import os
import subprocess
import zipfile
import time
import json
import nbformat
import requests
from git import Repo, GitCommandError, NoSuchPathError, InvalidGitRepositoryError
from loguru import logger
from paths import REPOS_DIR
from send2trash import send2trash
import pandas as pd
from pygments.lexers import guess_lexer_for_filename, TextLexer
from functools import wraps
from pygments.util import ClassNotFound
from token_count import num_tokens_from_string
from config import Config


def convert_ipynb_to_text(ipynb_content):
    notebook = json.loads(ipynb_content)
    text = ""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            text += ''.join(cell['source']) + '\n\n'
        elif cell['cell_type'] == 'code':
            text += '```python\n'
            text += ''.join(cell['source']) + '\n'
            text += '```\n\n'
            if len(cell['outputs']) > 0:
                text += '<output>\n'
                for output in cell['outputs']:
                    if output['output_type'] == 'stream':
                        text += ''.join(output['text']) + '\n'
                    elif output['output_type'] == 'execute_result':
                        text += ''.join(output['data'].get('text/plain', '')) + '\n'
                    elif output['output_type'] == 'error':
                        text += ''.join(output['traceback']) + '\n'
                text += '</output>\n\n'

    return text.strip()


def retry(max_retries=3, retry_delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None  # last exception that occurred
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (subprocess.CalledProcessError, requests.exceptions.RequestException, zipfile.BadZipFile) as e:
                    retries += 1
                    last_exception = e  # update last exception
                    logger.error(f"Error in {func.__name__}. Retrying ({retries}/{max_retries})...")
                    time.sleep(retry_delay)
            logger.error(f"Failed to execute {func.__name__} after {max_retries} retries.")
            if last_exception:
                raise last_exception  # if an exception occurred, raise it
            else:
                # usually this should not happen
                raise Exception(
                    f"Failed to execute {func.__name__} after {max_retries} retries without catching an exception.")

        return wrapper

    return decorator


class RepoService:
    def __init__(self, repo_url, repo_name=None):
        self.repo_url = repo_url
        self.repo_name = repo_name if repo_name else repo_url.split("/")[-1].replace(".git", "")
        self.repo_path = REPOS_DIR.joinpath(self.repo_name)
        self.clone_path = self.repo_path.joinpath(self.repo_name + "-main")
        self.clone_path.mkdir(exist_ok=True, parents=True)
        self.auth_key = os.getenv('GITHUB_TOKEN')
        self.repo = Repo(self.clone_path)
        self.current_branch = self.repo.active_branch.name

        if self.check_if_exist():
            logger.info(f"Repository {self.repo_name} already exists at {self.repo_path}")
        else:
            self.set_up()

    def get_branches(self):
        try:
            # Fetch all branches including remotes
            self.repo.git.fetch('--all')

            # Get local branches
            local_branches = [ref.name for ref in self.repo.heads]

            # Get remote branches and remove the 'origin/' prefix
            remote_branches = [ref.name.split('/', 1)[1] for ref in self.repo.remotes.origin.refs if '/' in ref.name]

            # Combine and remove duplicates
            all_branches = list(set(local_branches + remote_branches))

            # Sort branches alphabetically
            all_branches.sort()

            return all_branches
        except GitCommandError as e:
            logger.error(f"Failed to get branches: {e}")
            return []

    def check_if_exist(self):
        repo_info_path = self.repo_path.joinpath("repo_info.json")
        csv_path = self.repo_path.joinpath("repo_stats.csv")

        if not repo_info_path.exists() or not csv_path.exists():
            return False
        if pd.read_csv(csv_path).empty:
            return False

        with open(repo_info_path, "r") as f:
            repo_info = json.load(f)
            if "repo_url" not in repo_info or repo_info["repo_url"] != self.repo_url:
                return False

        # check if the repo has file, if no files, then return False
        if not os.listdir(self.clone_path):
            return False
        return True

    def set_up(self):
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path, exist_ok=True)
        repo_info = {"repo_url": self.repo_url}
        with open(self.repo_path.joinpath("repo_info.json"), "w") as f:
            json.dump(repo_info, f)
        self.clone_repo()
        if not self.repo_path.joinpath("repo_stats.csv").exists():
            self.get_repo_stats()
        logger.info(
            f"Repository {self.repo_name} set up successfully at {self.repo_path}")
        logger.info(
            f"Last updated: {time.ctime(os.path.getmtime(self.repo_path.joinpath('repo_stats.csv')))}")

    def clone_repo(self):
        if os.path.exists(self.clone_path) and os.listdir(self.clone_path):
            logger.info(f"The repository {self.repo_name} already exists at {self.clone_path}.")
            return True

        os.makedirs(self.clone_path, exist_ok=True)
        download_method = Config.get("download_method", "auto").lower()

        if download_method == "git":
            return self.try_clone_using_git()
        elif download_method == "http":
            return self.try_clone_using_http()
        elif download_method == "auto":
            if self.try_clone_using_git():
                return True
            logger.info("Git clone failed. Trying HTTP download.")
            return self.try_clone_using_http()

        logger.error(f"Invalid download method specified in config: {download_method}")
        return False

    def try_clone_using_git(self):
        try:
            subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._clone_using_git()
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to clone repository {self.repo_name} using Git. {e}")
            self.delete_repo()
            return False

    def try_clone_using_http(self):
        try:
            self._clone_using_download()
            return True
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            logger.error(f"Failed to clone repository {self.repo_name} using HTTP download method. {e}")
            self.delete_repo()
            return False

    @retry(max_retries=1, retry_delay=5)
    def _clone_using_git(self):
        logger.info(f"Cloning repository {self.repo_name} using Git...")
        if self.auth_key:
            # Use HTTPS with token authentication
            auth_url = self.repo_url.replace("https://", f"https://{self.auth_key}@")
            subprocess.run(["git", "clone", auth_url, self.clone_path], check=True, timeout=60)
        else:
            subprocess.run(["git", "clone", self.repo_url, self.clone_path], check=True, timeout=60)

    @retry(max_retries=1, retry_delay=5)
    def _clone_using_download(self):
        logger.info(f"Cloning repository {self.repo_name} using download...")
        headers = {}
        if self.auth_key:
            headers['Authorization'] = f'token {self.auth_key}'

        response = requests.get(self.repo_url, headers=headers, timeout=60)
        if response.status_code == 200:
            with open(os.path.join(self.repo_path, "repo.zip"), "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(os.path.join(self.repo_path, "repo.zip"), "r") as zip_ref:
                zip_ref.extractall(self.repo_path)
            os.remove(os.path.join(self.repo_path, "repo.zip"))
        else:
            raise requests.exceptions.RequestException(f"Failed to download repository {self.repo_name}")


    def delete_repo(self):
        if os.path.exists(self.repo_path):
            send2trash(self.repo_path)
            logger.info(
                f"Deleted repository {self.repo_name} at {self.repo_path}")
            return True
        else:
            logger.info(
                f"Repository {self.repo_name} does not exist at {self.repo_path}")
            return False

    def get_repo_stats(self):
        data = []
        for root, dirs, files in os.walk(self.clone_path):
            if '.git' in dirs:
                dirs.remove('.git')  # don't visit .git directories

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.clone_path)
                content = ''
                language = None
                if file.endswith('.ipynb'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook = nbformat.read(f, as_version=4)
                        content = nbformat.writes(notebook)
                        language = 'Jupyter Notebook'
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    try:
                        lexer = guess_lexer_for_filename(file_path, content)
                        language = lexer.name
                    except ClassNotFound:
                        language = None

                    if language is not None and isinstance(lexer, TextLexer):
                        language = None

                data.append({
                    'file_content': content,
                    'language': language,
                    'line_count': len(content.split('\n')),
                    'file_size': os.path.getsize(file_path),
                    'file_name': file,
                    'file_path': rel_path,
                    'token_count': num_tokens_from_string(content),
                    'description': None,
                    'graph': None
                })

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df.to_csv(csv_path, index=False, escapechar='\\')
        logger.info(f"Saved repo stats to {csv_path}")
        return df

    def filter_files(self, selected_files=None, selected_folders=None, selected_languages=None):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)
        df['file_path'] = df['file_path'].apply(
            lambda x: x.replace(os.sep, '/').replace('\\', '/').lower())

        final_condition = pd.Series([False] * len(df))

        if selected_files:
            selected_files = [path.replace(
                os.sep, '/').replace('\\', '/').lower() for path in selected_files]
            final_condition |= df['file_path'].isin(selected_files)

        if selected_folders:
            selected_folders = [folder.replace(
                os.sep, '/').replace('\\', '/').lower() for folder in selected_folders]
            folder_condition = pd.Series([any(df['file_path'].iloc[i].startswith(
                folder) for folder in selected_folders) for i in range(len(df))])
            final_condition |= folder_condition

        df = df[final_condition]

        if selected_languages:
            df = df[df['language'].isin(selected_languages)]

        return df

    def get_language_percentage(self):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)

        if df['language'].isna().all():
            logger.warning(
                "Warning: 'language' column is empty. Please make sure the 'language' column is populated.")
            return None

        language_counts = df.groupby('language')['line_count'].sum()
        total_lines = language_counts.sum()

        if total_lines == 0:
            logger.warning(
                "Warning: Total line count is zero. Cannot calculate language percentage.")
            return None

        language_percentage = language_counts / total_lines * 100
        return language_percentage

    def print_directory_structure(self):
        directory_structure = {}

        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.relpath(
                    os.path.join(root, file), self.repo_path)
                parts = file_path.split(os.sep)
                current_level = directory_structure

                for part in parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]

        def print_structure(structure, level=0):
            for key, value in structure.items():
                logger.info('  ' * level + '- ' + key)
                print_structure(value, level + 1)

        print_structure(directory_structure)

    def preprocess_dataframe(self, df, limit=None, concat_method='xml', include_directory=True, metadata_list=None):
        result = ''

        if include_directory:
            directory_structure = {}
            for _, row in df.iterrows():
                file_path = row['file_path']
                parts = file_path.split('/')
                current_level = directory_structure
                for part in parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]

            def flatten_directory(structure, prefix=''):
                flattened = []
                for key, value in structure.items():
                    flattened.append(prefix + key)
                    flattened.extend(flatten_directory(value, prefix + '  '))
                return flattened

            directory_lines = flatten_directory(directory_structure)
            result += 'Directory Structure:\n' + \
                      '\n'.join(directory_lines) + '\n\n'

        for _, row in df.iterrows():
            r = result
            result += '\n\n' + '=' * 10 + '\n\n'
            content = row['file_content']
            if row['language'] == 'Jupyter Notebook':
                content = convert_ipynb_to_text(content)

            if metadata_list:
                metadata = [str(row[col]) for col in metadata_list]
            else:
                metadata = ""

            if concat_method == 'xml':
                result += f'<file name="{row["file_path"]}">\n'
                if metadata:
                    result += f'<metadata>{", ".join(metadata)}</metadata>\n'
                result += f'<content>\n{content}\n</content>\n'
                result += '</file>'
            else:
                result += f'File: {row["file_path"]}\n'
                if metadata:
                    result += f'Metadata: {", ".join(metadata)}\n'
                result += f'Content:\n{content}'
            result += '\n\n' + '=' * 10 + '\n\n'
            if limit and num_tokens_from_string(result) > limit:
                result = r
                break

        return result.strip()

    def get_filtered_files(self, selected_folders=None, selected_files=None, selected_languages=None, limit=None,
                           concat_method='xml', include_directory=True, metadata_list=None):
        filtered_files = self.filter_files(
            selected_folders=selected_folders, selected_files=selected_files, selected_languages=selected_languages)
        file_string = self.preprocess_dataframe(filtered_files, limit=limit, concat_method=concat_method,
                                                include_directory=include_directory, metadata_list=metadata_list)
        return file_string

    def get_content_from_file_name(self, file_name):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)
        df = df[df["file_name"] == file_name]
        row = df.iloc[0]
        return row["file_content"]

    def get_folders_options(self):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)
        file_paths = df['file_path'].dropna().unique()
        # filter out files start with .git
        file_paths = [
            file for file in file_paths if not file.startswith('.git')]
        folders = list(set([os.path.dirname(file) for file in file_paths]))
        return sorted(folders)

    def get_files_options(self):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)
        # filter out files start with .git
        files = df['file_path'].dropna().unique()
        files = [file for file in files if not file.startswith('.git')]
        return sorted(files)

    def get_languages_options(self):
        csv_path = os.path.join(self.repo_path, "repo_stats.csv")
        df = pd.read_csv(csv_path)
        languages = df['language'].dropna().unique()
        return sorted(languages)

    def switch_branch(self, branch_name):
        try:
            if branch_name not in self.repo.heads:
                self.repo.git.checkout('-b', branch_name, f'origin/{branch_name}')
            else:
                self.repo.git.checkout(branch_name)

            self.current_branch = branch_name
            self.get_repo_stats()
            logger.info(f"Switched to branch: {branch_name}")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False

    def update_repo(self):
        try:
            logger.info(f"Updating repository {self.repo_name}...")
            origin = self.repo.remotes.origin

            if self.auth_key:
                # Set the remote URL with the auth key
                auth_url = self.repo_url.replace("https://", f"https://{self.auth_key}@")
                origin.set_url(auth_url)

            origin.fetch()
            current_commit = self.repo.head.commit
            remote_commit = origin.refs[self.current_branch].commit

            if current_commit.hexsha == remote_commit.hexsha:
                logger.info(f"Repository {self.repo_name} is already up-to-date.")
                return True

            origin.pull()
            logger.info(f"Repository {self.repo_name} updated successfully.")

            self.get_repo_stats()
            return True
        except GitCommandError as e:
            logger.error(f"Failed to update repository {self.repo_name}: {e}")
            return False


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class RepoManager:
    def __init__(self):
        logger.info("Initializing RepoManager...")
        self.repos = {}
        if not os.path.exists(REPOS_DIR):
            os.makedirs(REPOS_DIR, exist_ok=True)
        self.load_repos()
        logger.info(f"Loaded {len(self.repos)} repositories.")

    def _find_repos(self):
        repos = []
        top_level = REPOS_DIR
        for repo_dir in os.listdir(top_level):
            repo_path = os.path.join(top_level, repo_dir)
            if os.path.isdir(repo_path):
                if 'repo_stats.csv' in os.listdir(repo_path):
                    root = repo_path
                    repo_info_path = os.path.join(root, "repo_info.json")
                    repo_url_txt_path = os.path.join(root, "repo_url.txt")

                    if os.path.exists(repo_info_path):
                        with open(repo_info_path, "r") as f:
                            # logger.info(f"Reading repo info from {repo_info_path}")
                            try:
                                repo_info = json.load(f)
                                repo_url = repo_info.get(
                                    "repo_url", "").strip('"')
                                # fix repo_url if it has extra quotes
                                repo_info['repo_url'] = repo_url
                                with open(repo_info_path, "w") as f_update:
                                    json.dump(repo_info, f_update)
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Error decoding JSON from {repo_info_path}: {e}")
                    elif os.path.exists(repo_url_txt_path):
                        with open(repo_url_txt_path, "r") as f:
                            repo_url = f.read().strip().strip('"')  # legacy support
                        repo_info = {"repo_url": repo_url}
                        with open(repo_info_path, "w") as f:
                            json.dump(repo_info, f)
                        os.remove(repo_url_txt_path)  # delete legacy file
                    else:
                        repo_url = ""

                    if repo_url:
                        repos.append({
                            "repo_name": os.path.basename(root),
                            "repo_url": repo_url,
                            "last_updated": time.ctime(os.path.getmtime(os.path.join(root, "repo_stats.csv")))
                        })

        return repos

    def load_repos(self):
        repo_details = self._find_repos()
        for repo in repo_details:
            repo_url = repo["repo_url"]
            repo_name = repo["repo_name"]
            self.repos[repo_url] = RepoService(
                repo_url=repo_url, repo_name=repo_name)

    def add_repo(self, repo_url):
        if repo_url not in self.repos:
            repo_service = RepoService(repo_url=repo_url)
            if repo_service.check_if_exist():
                self.repos[repo_url] = repo_service
                logger.info(f"Added repository: {repo_url}")
            else:
                logger.error(f"Failed to add repository: {repo_url}")
                return False
        else:
            logger.warning(f"Repository already exists: {repo_url}")
        return True

    def delete_repo(self, repo_url):
        if repo_url in self.repos:
            self.repos[repo_url].delete_repo()
            del self.repos[repo_url]
            logger.info(f"Deleted repository: {repo_url}")
        else:
            logger.warning(f"Repository does not exist: {repo_url}")

    def update_all_repos(self):
        for repo_service in self.repos.values():
            repo_service.update_repo()

    def get_repo_service(self, repo_url) -> RepoService:
        return self.repos.get(repo_url)

    def get_repo_urls(self):
        return list(self.repos.keys())

    def check_if_repo_exists(self, repo_url):
        return repo_url in self.repos

    def isEmpty(self):
        return len(self.repos) == 0


if __name__ == "__main__":
    # repo_url -> reposervice
    repoManager = RepoManager()
    # repo = RepoService("https://github.com/jw782cn/RepoChat-200k")
    # repo.get_folders()
    # print(len(repo.get_folders()))
    # repo.delete_repo()
