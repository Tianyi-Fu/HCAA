
import os
from config.config import SHOW_CHANGED_HOLDS_OUTPUT, SHOW_CHANGED_HOLDS_NAME_OUTPUT
from asp.file_manager import (
    add_goal_to_asp_file,
    add_predicted_goal_to_asp_file,
    clean_existing_goals_and_success
)
from asp.solution_finder import run_sparc_solution_finder

def execute_user_goal(user_goal):

    clean_existing_goals_and_success()

    add_goal_to_asp_file(user_goal)


    predicates_to_show = ["occurs","show_operated_holds_name","show_changed_holds",
        "show_changed_holds_name","show_start_holds","show_last_holds"]
    run_sparc_solution_finder(display_predicates=predicates_to_show)

    rename_changed_holds_files("_user")


def execute_predicted_goal(predicted_goal):

    add_predicted_goal_to_asp_file(predicted_goal)

    full_predicates = [
        "occurs","show_operated_holds_name","show_changed_holds",
        "show_changed_holds_name","show_start_holds","show_last_holds"
    ]
    run_sparc_solution_finder(display_predicates=full_predicates)


    rename_changed_holds_files("_predicted")


def rename_changed_holds_files(suffix="_user"):

    old_changed = SHOW_CHANGED_HOLDS_OUTPUT
    new_changed = SHOW_CHANGED_HOLDS_OUTPUT.replace(".txt", f"{suffix}.txt")

    old_changed_name = SHOW_CHANGED_HOLDS_NAME_OUTPUT
    new_changed_name = SHOW_CHANGED_HOLDS_NAME_OUTPUT.replace(".txt", f"{suffix}.txt")


    if os.path.exists(new_changed):
        os.remove(new_changed)
    if os.path.exists(new_changed_name):
        os.remove(new_changed_name)


    if os.path.exists(old_changed):
        os.rename(old_changed, new_changed)

    if os.path.exists(old_changed_name):
        os.rename(old_changed_name, new_changed_name)
