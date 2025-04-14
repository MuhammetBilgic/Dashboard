from flask import Flask, render_template, request, Response
import pandas as pd
import MySQLdb
from math import ceil
from config import DB_CONFIG, AWS_CONFIG
from datetime import datetime
from functools import lru_cache
import boto3
import io

app = Flask(__name__)

# ---------------------------------------------------------------------
# AWS S3 Connection
# ---------------------------------------------------------------------

s3_client = boto3.client('s3', aws_access_key_id=AWS_CONFIG['aws_access_key_id'],
                         aws_secret_access_key=AWS_CONFIG['aws_secret_access_key'],
                         region_name=AWS_CONFIG['region_name'])


def read_csv_from_s3(key):
    response = s3_client.get_object(Bucket=AWS_CONFIG['s3_bucket'], Key=key)
    data = response['Body'].read()
    return pd.read_csv(io.BytesIO(data))

# ---------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------


def get_db_connection():
    # Use a context manager if desired, but here we simply return the connection.
    conn = MySQLdb.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        passwd=DB_CONFIG['password'],
        db=DB_CONFIG['database']
    )
    return conn

# ---------------------------------------------------------------------
# CSV Data Load Functions with Caching
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_annotations():
    # Change this with real path #######
    return read_csv_from_s3("data/annotations.csv")


@lru_cache(maxsize=1)
def load_time_metrics():
    return read_csv_from_s3("data/time_metrics.csv")


@lru_cache(maxsize=1)
def load_preventive_rw():
    return read_csv_from_s3("data/forward.csv")

# ---------------------------------------------------------------------
# Dynamic Page Range Function
# ---------------------------------------------------------------------


def get_dynamic_page_range(current_page, total_pages, delta=2):
    pages = []
    for i in range(1, total_pages + 1):
        if i == 1 or i == total_pages or abs(i - current_page) <= delta:
            pages.append(i)
        else:
            if pages and pages[-1] != '...':
                pages.append('...')
    return pages
# ---------------------------------------------------------------------
# Functions to Compute Metrics
# ---------------------------------------------------------------------


def compute_labelling_team_metrics(week, annotations, audits_df, filtered_time_metrics):
    # Get node-specific user IDs for the week
    user_ids = filtered_time_metrics.loc[filtered_time_metrics["amazon_week"]
                                         == week, "user_id"].unique()
    audit3 = int(annotations.loc[(annotations["amazon_week"] == week) & (
        annotations["annotator_id"].isin(user_ids)), "sum_of_asin"].sum())
    audit2 = int(audits_df.loc[(audits_df["amazon_week"] == week) & (
        audits_df["process"] == "CC audit") & (audits_df["uploader"].isin(user_ids)), "total_asin"].sum())
    audit1 = int(audits_df.loc[(audits_df["amazon_week"] == week) & (
        audits_df["process"] == "AMAS audit") & (audits_df["uploader"].isin(user_ids)), "total_asin"].sum())
    total = audit3 + audit2 + audit1
    time_spent = round(filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Refinement - Labeling"), "time_spent"].sum(), 2)
    tph = round(total / time_spent, 2) if time_spent > 0 else 0
    headcount = int(filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Refinement - Labeling"), "user_id"].nunique())
    return {"Audit3 Output": audit3, "Audit2 Output": audit2, "Audit1 Output": audit1, "Total Output": total, "Time Spent": time_spent, "TPH": tph, "Headcount": headcount}


def compute_labelling_individual_metrics(week, annotations, audits_df, filtered_time_metrics, selected_week="", selected_user=""):
    # Get node-specific user IDs for the week
    user_ids = filtered_time_metrics.loc[filtered_time_metrics["amazon_week"]
                                         == week, "user_id"].unique()
    ann = annotations.loc[(annotations["amazon_week"] == week) & (annotations["annotator_id"].isin(user_ids)), [
        "annotator_id", "sum_of_asin"]].rename(columns={"annotator_id": "user_id", "sum_of_asin": "total_asin"})
    aud = audits_df.loc[(audits_df["amazon_week"] == week) & (audits_df["uploader"].isin(
        user_ids)), ["uploader", "total_asin"]].rename(columns={"uploader": "user_id"})
    df = pd.concat([ann, aud])
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("user_id").sum().reset_index()
    user_time = filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Refinement - Labeling")].groupby("user_id")["time_spent"].sum().reset_index()
    merged = grouped.merge(user_time, on="user_id", how="left").fillna(0)
    merged["total_asin"] = merged["total_asin"].astype(int)
    merged["TPH"] = merged["total_asin"] / merged["time_spent"]
    merged["TPH"] = merged["TPH"].replace([float("inf"), -float("inf")], 0)
    merged["Week"] = week
    if selected_week:
        merged = merged[merged["Week"].astype(str) == selected_week]
    if selected_user:
        merged = merged[merged["user_id"].astype(str) == selected_user]
    return merged


def compute_writing_team_metrics(week, forward_data, filtered_time_metrics):
    user_ids = filtered_time_metrics.loc[filtered_time_metrics["amazon_week"]
                                         == week, "user_id"].unique()
    forward_out = int(forward_data.loc[(forward_data["amazon_week"] == week) & (
        forward_data["resolver"].isin(user_ids)), "asin_value"].sum())
    time_spent = round(filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Rule Writing"), "time_spent"].sum(), 2)
    headcount = int(filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Rule Writing"), "user_id"].nunique())
    filtered_forward = forward_data.loc[(forward_data["amazon_week"] == week) & (
        forward_data["resolver"].isin(user_ids))]
    cadence = filtered_forward["asin_value"].apply(
        lambda x: ceil(x/750) if x > 750 else 1).sum()
    tph = round(cadence/time_spent, 2) if time_spent > 0 else 0
    return {"Forward Output": forward_out, "Cadence": cadence, "Time Spent": time_spent, "TPH": tph, "Headcount": headcount}


def compute_writing_individual_metrics(week, forward_data, filtered_time_metrics, selected_week="", selected_user=""):
    user_ids = filtered_time_metrics.loc[filtered_time_metrics["amazon_week"]
                                         == week, "user_id"].unique()
    df = forward_data.loc[(forward_data["amazon_week"] == week) & (
        forward_data["resolver"].isin(user_ids))]
    if df.empty:
        return pd.DataFrame()
    group = df.groupby("resolver").agg(
        total_asin=("asin_value", "sum"),
        cadence=("asin_value", lambda x: sum(
            ceil(val/750) if val > 750 else 1 for val in x))
    ).reset_index().rename(columns={"resolver": "user_id"})
    user_time = filtered_time_metrics.loc[(filtered_time_metrics["amazon_week"] == week) & (
        filtered_time_metrics["task_name"] == "BYOC Rule Writing")].groupby("user_id")["time_spent"].sum().reset_index()
    merged = group.merge(user_time, on="user_id", how="left").fillna(0)
    merged["total_asin"] = merged["total_asin"].astype(int)
    merged["TPH"] = merged["cadence"] / merged["time_spent"]
    merged["TPH"] = merged["TPH"].replace([float("inf"), -float("inf")], 0)
    merged["Week"] = week
    if selected_week:
        merged = merged[merged["Week"].astype(str) == selected_week]
    if selected_user:
        merged = merged[merged["user_id"].astype(str) == selected_user]
    return merged

# ---------------------------------------------------------------------
# Dashboard Endpoint / Ana Sayfa Endpoint'i
# ---------------------------------------------------------------------


@app.route('/')
def dashboard_endpoint():
    # Get filter parameters
    selected_tab = request.args.get("tab", "Labelling")
    selected_week = request.args.get("week", "")
    selected_user = request.args.get("user_id", "")
    selected_node = request.args.get("node_id", "")
    page = int(request.args.get("page", 1))
    per_page = 20

    annotations = load_annotations()
    time_metrics = load_time_metrics()
    forward_data = load_preventive_rw()

    # Filter time_metrics by node_id if provided
    if selected_node:
        filtered_time_metrics = time_metrics[time_metrics["node_id"].astype(
            str) == selected_node]
    else:
        filtered_time_metrics = time_metrics

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            DATE_FORMAT(upload_date, '%x-%v') AS amazon_week,
            uploader,
            COUNT(NULLIF(ASIN, '')) as total_asin,
            CASE
                WHEN stream REGEXP 'CC[0-9]{4}|.*WK[0-9]{2}' THEN 'CC audit'
                WHEN stream NOT LIKE '%cc%' THEN 'AMAS audit'
                ELSE 'Other'
            END AS process
        FROM audits
        WHERE stream IS NOT NULL
            AND stream != ''
        GROUP BY amazon_week, uploader, process
"""
    audits_data=cursor.fetchall()
    conn.close()
    audits_df=pd.DataFrame(audits_data, columns=[
                             "amazon_week", "uploader", "total_asin", "process"])

    weeks=sorted(filtered_time_metrics["amazon_week"].unique(), reverse=True)
    users=sorted(filtered_time_metrics["user_id"].unique())
    node_ids=sorted(time_metrics["node_id"].unique())

    team_labelling={}  # Initialize dictionary for team labelling metrics
    individual_labelling_list=[]
    for week in weeks:
        team_labelling[week]=compute_labelling_team_metrics(
            week, annotations, audits_df, filtered_time_metrics)
        ind_df=compute_labelling_individual_metrics(
            week, annotations, audits_df, filtered_time_metrics, selected_week, selected_user)
        if not ind_df.empty:
            individual_labelling_list.extend(ind_df.to_dict(orient="records"))

    team_preventive_rw={}  # Initialize dictionary for team writing metrics
    individual_preventive_rw=[]
    for week in weeks:
        team_preventive_rw[week]=compute_writing_team_metrics(
            week, forward_data, filtered_time_metrics)
        ind_df=compute_writing_individual_metrics(
            week, forward_data, filtered_time_metrics, selected_week, selected_user)
        if not ind_df.empty:
            individual_preventive_rw.extend(ind_df.to_dict(orient="records"))

    if selected_tab == "Labelling":
        total_items=len(individual_labelling_list)
        total_pages=ceil(total_items / per_page) if total_items > 0 else 1
        start_index=(page - 1) * per_page
        end_index=start_index + per_page
        individual_labelling_paginated=individual_labelling_list[start_index:end_index]
    else:
        total_items=len(individual_preventive_rw)
        total_pages=ceil(total_items / per_page) if total_items > 0 else 1
        start_index=(page - 1) * per_page
        end_index=start_index + per_page
        individual_writing_paginated=individual_preventive_rw[start_index:end_index]

    dynamic_page_range=get_dynamic_page_range(page, total_pages, delta=2)

    return render_template("main.html",
                           weeks=weeks,
                           users=users,
                           node_ids=node_ids,
                           selected_tab=selected_tab,
                           selected_week=selected_week,
                           selected_user=selected_user,
                           selected_node=selected_node,
                           current_page=page,
                           total_pages=total_pages,
                           dynamic_page_range=dynamic_page_range,
                           team_labelling=team_labelling,
                           individual_labelling=individual_labelling_paginated if selected_tab == "Labelling" else [],
                           team_preventive_rw=team_preventive_rw,
                           individual_writing=individual_writing_paginated if selected_tab == "Writing" else [])

# ---------------------------------------------------------------------
# Export CSV Endpoint
# ---------------------------------------------------------------------


@ app.route('/export')
def export_csv():
    selected_tab=request.args.get("tab", "Labelling")
    export_type=request.args.get("export_type", "individual")
    selected_week=request.args.get("week", "")
    selected_user=request.args.get("user_id", "")
    selected_node=request.args.get("node_id", "")

    annotations=load_annotations()
    time_metrics=load_time_metrics()
    forward_data=load_preventive_rw()

    if selected_node:
        filtered_time_metrics=time_metrics[time_metrics["node_id"].astype(
            str) == selected_node]
    else:
        filtered_time_metrics=time_metrics

    conn=get_db_connection()
    cursor=conn.cursor()
    cursor.execute("""
        SELECT
            DATE_FORMAT(upload_date, '%x-%v') AS amazon_week,
            uploader,
            COUNT(NULLIF(ASIN, '')) as total_asin,
            CASE
                WHEN stream REGEXP 'CC[0-9]{4}|.*WK[0-9]{2}' THEN 'CC audit'
                WHEN stream NOT LIKE '%cc%' THEN 'AMAS audit'
                ELSE 'Other'
            END AS process
        FROM audits
        WHERE stream IS NOT NULL
            AND stream != ''
        GROUP BY amazon_week, uploader, process
"""

    audits_data=cursor.fetchall()
    conn.close()
    audits_df=pd.DataFrame(audits_data, columns=[
                             "amazon_week", "uploader", "total_asin", "process"])
    weeks=sorted(filtered_time_metrics["amazon_week"].unique(), reverse=True)

    if export_type == "individual":
        if selected_tab == "Labelling":
            export_list=[]
            for week in weeks:
                node_user_ids=filtered_time_metrics.loc[filtered_time_metrics["amazon_week"] == week, "user_id"].unique(
                )
                individual_users=pd.concat([
                    annotations[
                        (annotations["amazon_week"] == week) &
                        (annotations["annotator_id"].isin(node_user_ids))
                    ][["annotator_id", "sum_of_asin"]].rename(
                        columns={"annotator_id": "user_id",
                                 "sum_of_asin": "total_asin"}
                    ),
                    audits_df[
                        (audits_df["amazon_week"] == week) &
                        (audits_df["uploader"].isin(node_user_ids))
                    ][["uploader", "total_asin"]].rename(
                        columns={"uploader": "user_id"}
                    )
                ])
                user_group=individual_users.groupby(
                    "user_id").sum().reset_index()
                user_time=filtered_time_metrics[
                    (filtered_time_metrics["amazon_week"] == week) &
                    (filtered_time_metrics["task_name"]
                     == "BYOC Refinement - Labeling")
                ].groupby("user_id")["time_spent"].sum().reset_index()
                user_metrics=user_group.merge(
                    user_time, on="user_id", how="left").fillna(0)
                user_metrics["total_asin"]=user_metrics["total_asin"].astype(
                    int)
                user_metrics["TPH"]=user_metrics["total_asin"] /
                    user_metrics["time_spent"]
                user_metrics["TPH"]=user_metrics["TPH"].replace(
                    [float("inf"), -float("inf")], 0)
                user_metrics["Week"]=week

                if selected_week:
                    user_metrics=user_metrics[user_metrics["Week"].astype(
                        str) == selected_week]
                if selected_user:
                    user_metrics=user_metrics[user_metrics["user_id"].astype(
                        str) == selected_user]

                export_list.extend(user_metrics.to_dict(orient="records"))
            export_df=pd.DataFrame(export_list)
        else:
            export_list=[]
            for week in weeks:
                node_user_ids=filtered_time_metrics.loc[filtered_time_metrics["amazon_week"] == week, "user_id"].unique(
                )
                user_metrics=forward_data[
                    (forward_data["amazon_week"] == week) &
                    (forward_data["resolver"].isin(node_user_ids))
                ].groupby("resolver").agg(
                    total_asin=("asin_value", "sum"),
                    cadence=("asin_value", lambda x: sum(
                        ceil(val/750) if val > 750 else 1 for val in x))
                ).reset_index().rename(columns={"resolver": "user_id"})
                user_time=filtered_time_metrics[
                    (filtered_time_metrics["amazon_week"] == week) &
                    (filtered_time_metrics["task_name"] == "BYOC Rule Writing")
                ].groupby("user_id")["time_spent"].sum().reset_index()
                user_metrics=user_metrics.merge(
                    user_time, on="user_id", how="left").fillna(0)
                user_metrics["total_asin"]=user_metrics["total_asin"].astype(
                    int)
                user_metrics["TPH"]=user_metrics["cadence"] /
                    user_metrics["time_spent"]
                user_metrics["TPH"]=user_metrics["TPH"].replace(
                    [float("inf"), -float("inf")], 0)
                user_metrics["Week"]=week

                if selected_week:
                    user_metrics=user_metrics[user_metrics["Week"].astype(
                        str) == selected_week]
                if selected_user:
                    user_metrics=user_metrics[user_metrics["user_id"].astype(
                        str) == selected_user]

                export_list.extend(user_metrics.to_dict(orient="records"))
            export_df=pd.DataFrame(export_list)
    elif export_type == "team":
        if selected_tab == "Labelling":
            team_list=[]
            for week in weeks:
                node_user_ids=filtered_time_metrics.loc[filtered_time_metrics["amazon_week"] == week, "user_id"].unique(
                )
                audit3_output=int(
                    annotations[
                        (annotations["amazon_week"] == week) &
                        (annotations["annotator_id"].isin(node_user_ids))
                    ]["sum_of_asin"].sum()
                )
                audit2_output=int(
                    audits_df[
                        (audits_df["amazon_week"] == week) &
                        (audits_df["process"] == "CC audit") &
                        (audits_df["uploader"].isin(node_user_ids))
                    ]["total_asin"].sum()
                )
                audit1_output=int(
                    audits_df[
                        (audits_df["amazon_week"] == week) &
                        (audits_df["process"] == "AMAS audit") &
                        (audits_df["uploader"].isin(node_user_ids))
                    ]["total_asin"].sum()
                )
                total_output=audit3_output + audit2_output + audit1_output
                time_spent=round(
                    filtered_time_metrics[
                        (filtered_time_metrics["amazon_week"] == week) &
                        (filtered_time_metrics["task_name"]
                         == "BYOC Refinement - Labeling")
                    ]["time_spent"].sum(), 2
                )
                tph=round(total_output / time_spent,
                            2) if time_spent > 0 else 0
                headcount=int(
                    filtered_time_metrics[
                        (filtered_time_metrics["amazon_week"] == week) &
                        (filtered_time_metrics["task_name"]
                         == "BYOC Refinement - Labeling")
                    ]["user_id"].nunique()
                )
                team_list.append({
                    "Week": week,
                    "Audit3 Output": audit3_output,
                    "Audit2 Output": audit2_output,
                    "Audit1 Output": audit1_output,
                    "Total Output": total_output,
                    "time_spent": time_spent,
                    "TPH": tph,
                    "Headcount": headcount
                })
            export_df=pd.DataFrame(team_list)
        else:
            team_list=[]
            for week in weeks:
                forward_output=int(
                    forward_data[forward_data["amazon_week"] == week]["asin_value"].sum())
                time_spent=round(
                    filtered_time_metrics[
                        (filtered_time_metrics["amazon_week"] == week) &
                        (filtered_time_metrics["task_name"]
                         == "BYOC Rule Writing")
                    ]["time_spent"].sum(), 2
                )
                headcount=int(
                    filtered_time_metrics[
                        (filtered_time_metrics["amazon_week"] == week) &
                        (filtered_time_metrics["task_name"]
                         == "BYOC Rule Writing")
                    ]["user_id"].nunique()
                )
                cadence=forward_data[forward_data["amazon_week"] == week]["asin_value"].apply(
                    lambda x: ceil(x / 750) if x > 750 else 1
                ).sum()
                tph=round(cadence / time_spent, 2) if time_spent > 0 else 0
                team_list.append({
                    "Week": week,
                    "Forward Output": forward_output,
                    "Cadence": cadence,
                    "time_spent": time_spent,
                    "TPH": tph,
                    "Headcount": headcount
                })
            export_df=pd.DataFrame(team_list)

    # --- Set column order & create dynamic filename with timestamp, export type and tab ---
    if export_type == "individual":
        if selected_tab == "Labelling":
            export_df=export_df[["Week", "user_id",
                                   "total_asin", "time_spent", "TPH"]]
        else:
            export_df=export_df[["Week", "user_id",
                                   "total_asin", "cadence", "time_spent", "TPH"]]
    elif export_type == "team":
        if selected_tab == "Labelling":
            export_df=export_df[["Week", "UTC Output", "CC Output",
                                   "AMAS Output", "Total Output", "time_spent", "TPH", "Headcount"]]
        else:
            export_df=export_df[["Week", "KW RW Output",
                                   "Cadence", "time_spent", "TPH", "Headcount"]]
    now=datetime.now().strftime("%Y%m%d_%H%M%S")
    filename=f"{selected_tab}_{export_type}_{now}.csv"

    csv_data=export_df.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename={filename}"})


if __name__ == '__main__':
    app.run(debug=True)
