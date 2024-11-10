import os
import pandas as pd
import streamlit as st
from loguru import logger
from openai import OpenAI
from token_count import num_messages, num_tokens_from_string
from llm_service import MODELS, create_client_for_model
from repo_service import RepoManager, RepoService

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class AppSettings:
    repo_url: str
    selected_branch: str
    selected_folders: List[str]
    selected_files: List[str]
    selected_languages: List[str]
    file_limit: int
    model: str
    temperature: float
    system_prompt: str


class SettingsManager:
    def __init__(self, settings_dir: str = "settings"):
        self.settings_dir = settings_dir
        os.makedirs(settings_dir, exist_ok=True)

    def save_settings(self, name: str, settings: AppSettings):
        """Save settings to a JSON file"""
        filepath = os.path.join(self.settings_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(settings), f, indent=2)

    def load_settings(self, name: str) -> Optional[AppSettings]:
        """Load settings from a JSON file"""
        filepath = os.path.join(self.settings_dir, f"{name}.json")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return AppSettings(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def list_settings(self) -> List[str]:
        """List all saved settings profiles"""
        files = os.listdir(self.settings_dir)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]

    def delete_settings(self, name: str) -> bool:
        """Delete a settings profile"""
        filepath = os.path.join(self.settings_dir, f"{name}.json")
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError:
            return False


class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def process_token(self, token: str):
        self.text += token
        self.container.markdown(self.text)


def refresh_repos():
    logger.info("Refreshing repositories")
    if 'repoManager' not in st.session_state:
        st.session_state['repoManager'] = RepoManager()
    st.session_state['repoManager'].load_repos()
    st.success("Refreshed repositories")


def clear_conversation():
    st.session_state["messages"] = []
    st.success("Conversation cleared")


def merge_with_available_options(saved_items: list, available_options: list) -> list:
    """Helper function to merge saved items with available options"""
    if not saved_items:
        return []
    # Keep only items that still exist in available options
    return [item for item in saved_items if item in available_options]


def create_app():
    st.set_page_config(page_title="ChatWithRepo", page_icon="🤖")

    if 'repoManager' not in st.session_state:
        st.session_state['repoManager'] = RepoManager()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if 'settings_manager' not in st.session_state:
        st.session_state['settings_manager'] = SettingsManager()

    repoManager: RepoManager = st.session_state['repoManager']
    settings_manager: SettingsManager = st.session_state['settings_manager']

    with st.sidebar:
        st.title("Settings Management")

        # First, get the repo URL selection since we need it for both settings and repo operations
        custom_repo_url = st.text_input("Custom Repository URL")
        if st.button("Add Custom Repository"):
            if repoManager.add_repo(custom_repo_url):
                st.success(f"Added custom repository: {custom_repo_url}")
            else:
                st.error(f"Repository add failed: {custom_repo_url}")

        if st.button("Refresh Repositories"):
            refresh_repos()

        # Get repository URL selection
        repo_url = st.selectbox(
            "Repository URL",
            options=repoManager.get_repo_urls(),
            index=0
        )

        # Now we can safely handle settings management
        settings_name = st.text_input("Settings Profile Name")
        if st.button("Save Current Settings"):
            if settings_name:
                repo = repoManager.get_repo_service(repo_url) if repoManager.check_if_repo_exists(repo_url) else None

                current_settings = AppSettings(
                    repo_url=repo_url,
                    selected_branch=repo.current_branch if repo else "",
                    selected_folders=st.session_state.get('default_folders', []),
                    selected_files=st.session_state.get('default_files', []),
                    selected_languages=st.session_state.get('default_languages', []),
                    file_limit=st.session_state.get('limit', 100000),
                    model=st.session_state.get('selected_model', MODELS[0]),
                    temperature=st.session_state.get('temperature', 0.7),
                    system_prompt=st.session_state.get('system_prompt', "")
                )
                settings_manager.save_settings(settings_name, current_settings)
                st.success(f"Saved settings profile: {settings_name}")

        # Load/Delete Settings
        saved_profiles = settings_manager.list_settings()
        if saved_profiles:
            selected_profile = st.selectbox("Saved Profiles", options=saved_profiles)
            if st.button("Load Selected Profile"):
                settings = settings_manager.load_settings(selected_profile)
                if settings:
                    st.session_state['load_settings'] = settings
                    st.success(f"Loaded settings: {selected_profile}")
                    st.rerun()
            if st.button("Delete Selected Profile"):
                if settings_manager.delete_settings(selected_profile):
                    st.success(f"Deleted settings: {selected_profile}")
                    st.rerun()

        # Apply loaded settings if they exist
        default_repo_url = None
        default_branch = None
        default_folders = None
        default_files = ["README.md"]
        default_languages = None
        default_limit = 100000
        default_model = MODELS[0]
        default_temperature = 0.7
        default_system_prompt = "You are a helpful assistant. You are provided with a repo information and files from the repo. Answer the user's questions based on the information and files provided."

        if 'load_settings' in st.session_state:
            settings = st.session_state.pop('load_settings')
            default_repo_url = settings.repo_url
            default_branch = settings.selected_branch
            default_folders = settings.selected_folders
            default_files = settings.selected_files
            default_languages = settings.selected_languages
            default_limit = settings.file_limit
            default_model = settings.model
            default_temperature = settings.temperature
            default_system_prompt = settings.system_prompt

            # Store these in session state to maintain them across reruns
            if 'default_folders' not in st.session_state:
                st.session_state.default_folders = default_folders
            if 'default_files' not in st.session_state:
                st.session_state.default_files = default_files
            if 'default_languages' not in st.session_state:
                st.session_state.default_languages = default_languages

        default_folders = st.session_state.get('default_folders', default_folders)
        default_files = st.session_state.get('default_files', default_files)
        default_languages = st.session_state.get('default_languages', default_languages)


        if repoManager.check_if_repo_exists(repo_url):
            repo = repoManager.get_repo_service(repo_url)

            st.subheader("Branch Selection")
            branches = repo.get_branches()
            selected_branch = st.selectbox(
                "Select Branch",
                options=branches,
                index=branches.index(default_branch) if default_branch in branches else (
                    branches.index(repo.current_branch) if repo.current_branch in branches else 0
                )
            )

            if selected_branch != repo.current_branch:
                if st.button(f"Switch to {selected_branch}"):
                    with st.spinner(f"Switching to branch: {selected_branch}"):
                        if repo.switch_branch(selected_branch):
                            st.success(f"Switched to branch: {selected_branch}")
                            st.info("Refreshing repository data...")
                            repo.get_repo_stats()
                            st.rerun()
                        else:
                            st.error(f"Failed to switch to branch: {selected_branch}")

        # Get available options
        available_folders = repo.get_folders_options()
        available_files = repo.get_files_options()
        available_languages = repo.get_languages_options()

        # Merge saved selections with available options
        merged_folders = merge_with_available_options(default_folders, available_folders)
        merged_files = merge_with_available_options(default_files, available_files)
        merged_languages = merge_with_available_options(default_languages, available_languages)

        selected_folder = st.multiselect(
            "Select Folder",
            options=repo.get_folders_options(),
            default=default_folders if default_folders else []
        )
        selected_files = st.multiselect(
            "Select Files",
            options=repo.get_files_options(),
            default=default_files if default_files else ["README.md"]
        )
        selected_languages = st.multiselect(
            "Filtered by Language",
            options=repo.get_languages_options(),
            default=default_languages if default_languages else []
        )

        st.session_state.default_folders = selected_folder
        st.session_state.default_files = selected_files
        st.session_state.default_languages = selected_languages

        limit = st.number_input("Limit", value=default_limit, step=10000)

        if st.button("Count Tokens"):
            file_string = repo.get_filtered_files(
                selected_folders=selected_folder,
                selected_files=selected_files,
                selected_languages=selected_languages,
                limit=limit,
            )
            st.write(f"Total Tokens: {num_tokens_from_string(file_string)}")

        if st.button("Update Repo"):
            if repo.update_repo():
                st.success(f"Updated repository: {repo_url}")
            else:
                st.error(f"Repository update failed: {repo_url}")
            st.rerun()

        if st.button("Delete Repo"):
            if repo.delete_repo():
                st.success(f"Deleted repository: {repo_url}")
            else:
                st.error(f"Repository delete failed: {repo_url}")
            refresh_repos()
            st.rerun()

        # LLM Settings Section
        st.title("Settings for LLM")
        selected_model = st.selectbox(
            "Model",
            options=MODELS,
            index=MODELS.index(default_model) if default_model in MODELS else 0
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=default_temperature,
            step=0.1
        )
        system_prompt = st.text_area(
            "System Prompt",
            value=default_system_prompt
        )

    # Rest of your existing app code remains the same
    if "client" not in st.session_state:
        st.session_state["client"] = create_client_for_model(selected_model)

    if repoManager.isEmpty():
        st.info("Copy the repository URL and click the download button.")
        st.stop()

    if not repoManager.check_if_repo_exists(repo_url):
        st.info(f"{repo_url} does not exist. Please add the repository first.")
        st.stop()

    repo = repoManager.get_repo_service(repo_url)
    st.title(f"Repo: {repo.repo_name} (Branch: {repo.current_branch})")

    # Add Clear Conversation button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(
            "Chat with LLM using the repository information and files. You can change model settings anytime during the chat."
        )
    with col2:
        if st.button("Clear Conversation"):
            clear_conversation()
            st.rerun()

    st.info(
        f"""
    Files : {selected_files}
    Folder: {selected_folder}
    Languages: {selected_languages}
    Limit: {limit}
    """
    )
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        logger.info(f"User: {prompt}, received at {pd.Timestamp.now()}")

        start_time = pd.Timestamp.now()
        # Check if the selected model has changed
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

        if st.session_state.selected_model != selected_model:
            st.session_state.client = create_client_for_model(selected_model)
            st.session_state.selected_model = selected_model

        file_string = repo.get_filtered_files(
            selected_folders=selected_folder,
            selected_files=selected_files,
            selected_languages=selected_languages,
            limit=limit,
        )
        end_time = pd.Timestamp.now()
        logger.info(f"Time taken to get filtered files: {end_time - start_time}")

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            messages = (
                    [{"role": "system", "content": system_prompt}]
                    + [{"role": "user", "content": file_string}]
                    + st.session_state.messages
            )
            client = st.session_state["client"]

            total_tokens = num_messages(messages)
            logger.info(f"Information: {selected_files}, {selected_folder}, {selected_languages}")
            logger.info(f"Using settings: {selected_model}, {temperature}")
            logger.info(f"File token: {num_tokens_from_string(file_string)}")
            logger.info(f"Total Messages Token: {total_tokens}")
            st.sidebar.write(
                f"Sending file content: {selected_files} and filter folder: {selected_folder} to the assistant.")
            st.sidebar.write(f"total messages token: {total_tokens}")

            completion = client.chat(
                messages, stream=True, temperature=temperature, model=selected_model
            )

            for chunk in completion:
                content = chunk.choices[0].delta.content
                stream_handler.process_token(content)

            st.session_state.messages.append(
                {"role": "assistant", "content": stream_handler.text}
            )


if __name__ == "__main__":
    create_app()
