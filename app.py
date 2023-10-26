import os
from h2o_wave import main, app, Q, ui
import shutil  ##

# Set the path to the images folder
IMAGES_FOLDER = '/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/images'

# Create the images folder if it doesn't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# -----------------------------------------------
global result
global number
from datetime import datetime
import os
import glob
import json
import dill
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from hydrogen_torch.src.models.convert_3d import convert_3d
from hydrogen_torch.src.utils.data_utils import get_inference_batch_size
from hydrogen_torch.src.utils.modeling_utils import load_checkpoint, run_python_scoring_inference

def delete_old_files(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Check the number of files
    if len(files) <= 1:
        print(f"There are {len(files)} file(s) in the folder. No files will be deleted.")
        return

    # Get the creation time of each file
    file_times = [(f, os.path.getctime(os.path.join(folder_path, f))) for f in files]

    # Sort files by creation time in descending order
    file_times.sort(key=lambda x: x[1], reverse=True)

    # Keep the latest file and delete the rest
    for file, _ in file_times[1:]:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

    print(f"Kept the latest file: {os.path.join(folder_path, file_times[0][0])}")

def run_scoring_pipeline():
    # Define the folder path
    folder_path = "/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/images"

    # Keep only the latest file
    delete_old_files(folder_path)

    # Reading the config from trained experiment
    with open("cfg.p", "rb") as pickle_file:
        cfg = dill.load(pickle_file)

    # Changing internal cfg settings for inference, not subject to change
    cfg.prediction._calculate_test_metric = False

    # Preparing exemplary dataframe for inference loading samples
    # This has to be altered for custom data

    # Image data -------------------------------------------------------
    if hasattr(cfg.dataset, "image_column"):
        images = []
        for image in sorted(glob.glob("images/*")):
            images.append(os.path.basename(image))

        test_df = pd.DataFrame({f"{cfg.dataset.image_column}": images})

        # Set image folder
        cfg.dataset.data_folder_test = "images"
    # -----------------------------------------------------------------
    #     test_df = pd.DataFrame.from_dict(questions_and_contexts)
    # # ------------------------------------------------------------------

    # Set device for inference
    if torch.cuda.is_available():
        cfg.environment._device = "cuda"
    else:
        cfg.environment._device = "cpu"

    # Disable original pretrained weights for model initialization
    if hasattr(cfg.architecture, "pretrained"):
        cfg.architecture.pretrained = False

    # It is possible to specify a custom cache directory for Huggingface models
    if hasattr(cfg, "transformers_cache_directory"):
        cfg.transformers_cache_directory = None

    # Loading model and checkpoint
    model = cfg.architecture.model_class(cfg).eval().to(cfg.environment._device)

    # Convert to 3D CNNs if needed
    if hasattr(cfg.architecture, "is_3d") and cfg.architecture.is_3d:
        model = convert_3d(model)

    cfg.architecture.pretrained_weights = "checkpoint.pth"
    load_checkpoint(cfg, model)

    # Preparing torch dataset and dataloader
    # Batch_size and num_workers are subject to change
    batch_size = get_inference_batch_size(cfg)

    test_dataset = cfg.dataset.dataset_class(df=test_df, cfg=cfg, mode="test")

    # Use num_workers=0 to avoid pickling issues
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_df),
        batch_size=batch_size,
        num_workers=0,  # Set to 0 to use a single worker process
        pin_memory=True,
        collate_fn=test_dataset.get_validation_collate_fn(),
    )

    # Running actual inference
    # Raw_predictions is a dictionary with predictions in the raw format
    # df_predictions is a pandas DataFrame with predictions
    raw_predictions, df_predictions = run_python_scoring_inference(
        cfg=cfg, model=model, dataloader=test_dataloader
    )

    # Final output
    print(df_predictions.head())
    result = df_predictions['pred_label'].iloc[0]
    print(result)
    return result

#------------------------------------------------
#make it asynchronous
import os
from h2o_wave import main, app, Q, ui, AsyncSite
import shutil
import asyncio  # Import asyncio module

#-------------------------------------------------
# Define the Wave app
persona = 'https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&h=750&w=1260'

@app('/')
async def serve(q: Q):
#----------
    if 'file_upload' in q.args:
        uploaded_file = q.args.file_upload[0]
        file_to_copy = '/home/ladmin/inspiredemo/waveserver/venvwave/data/' + uploaded_file[2:]
        shutil.copy(file_to_copy, IMAGES_FOLDER)

        q.page['example'] = ui.form_card(box='1 1 5 6', items=[
            ui.text(f'Uploaded image saved with the original filename: {uploaded_file}'),
            ui.button(name='show_upload', label='Back', primary=True),
            ui.image(title=file_to_copy, path=uploaded_file)
        ])

        # Keep only the latest file
        delete_old_files("/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/images")

        # Display "example2" card only when score() has finished processing
        q.page['example2'] = ui.markdown_card(
            box='6 1 5 1',
            title='Prediction',
            content='Processing...',
        )
#--------------
        from h2o_wave import data
        q.page['example8'] = ui.plot_card(
            box='6 8 5 6',
            title='Heatmap',
            data=data('year person sales', 10, rows=[
                ('Zone 1', 'East Wing', 10),
                ('Zone 2', 'West Wing', 58),
                ('Zone 3', 'North Wing', 114),
                ('Zone 1', 'South Wing', 31),
                ('Zone 3', 'East Wing', 96),
                ('Zone 1', 'West Wing', 55),
                ('Zone 3', 'West Wing', 5),
                ('Zone 2', 'South Wing', 85),
                ('Zone 3', 'South Wing', 132),
                ('Zone 2', 'East Wing', 54),
                ('Zone 1', 'North Wing', 78),
                ('Zone 2', 'North Wing', 18),
            ]),
            plot=ui.plot([ui.mark(type='polygon', x='=person', y='=year', color='=sales',
                        color_range='#fee8c8 #fdbb84 #e34a33')])
        )
#-----------------
        q.page['example5'] = ui.wide_pie_stat_card(
            box='6 3 2 3',
            title='Admissions today',
            pies=[
                ui.pie(label='Fyre Festival', value='35%', fraction=0.35, color='#2cd0f5', aux_value='35'),
                ui.pie(label='Sundown Event', value='65%', fraction=0.65, color='$themePrimary', aux_value='65'),
            ],
        )
#------------------
        q.page['example9'] = ui.article_card(
            box='6 6 2 2',
            title='Updates',
            items=[
                ui.mini_buttons([
                    ui.mini_button(name='like', label='4', icon='Heart'),
                    ui.mini_button(name='comment', label='2', icon='Blog'),
                    ui.mini_button(name='share', label='1', icon='Relationship'),
                ])
            ],
            content='''
        Duis porttitor tincidunt justo ac semper. Vestibulum et molestie lectus. Proin vel eros a ex condimentum aliquam.
        Sed accumsan tellus sit amet nulla ullamcorper. Suspendisse bibendum tristique sem, quis lacinia ex pulvinar quis.
        Nam elementum accumsan porta. Sed eget aliquam elit, sed luctus lorem. Nulla gravida malesuada purus eu eleifend.
        Maecenas in ante interdum, hendrerit velit at, tempus eros. Nullam convallis tempor libero at viverra.

        ## Heading 2

        Duis porttitor tincidunt justo ac semper. Vestibulum et molestie lectus. Proin vel eros a ex condimentum aliquam.
        Sed accumsan tellus sit amet nulla ullamcorper. Suspendisse bibendum tristique sem, quis lacinia ex pulvinar quis.
        Nam elementum accumsan porta. Sed eget aliquam elit, sed luctus lorem. Nulla gravida malesuada purus eu eleifend.
        Maecenas in ante interdum, hendrerit velit at, tempus eros. Nullam convallis tempor libero at viverra.
        '''
        )
#-----------------
        q.page['example6'] = ui.form_card(box='8 3 3 2', items=[
        ui.stats([
            ui.stat(label='Cat A', value='$ 60', caption='Cat A', icon='Home'),
            ui.stat(label='Cat B', value='$ 120', caption='Cat B', icon='Cake'),
            ui.stat(label='Cat C', value='$ 300', caption='Cat C', icon='Heart'),
        ])
    ])
#------------------------------------
        from h2o_wave import data
        q.page['example7'] = ui.plot_card(
            box='8 5 3 3',
            title='Total entrants in the last hour',
            data=data('month price', 12, rows=[
                ('East Wing 1', 51),
                ('East Wing 2', 91),
                ('East Wing 3', 34),
                ('North Wing 1', 47),
                ('North Wing 2', 63),
                ('North Wing 3', 58),
                ('South Wing 1', 56),
                ('South Wing 2', 77),
                ('South Wing 3', 99),
                ('West Wing 1', 106),
                ('West Wing 2', 88),
                ('West Wing 3', 56),
            ]),
            plot=ui.plot([
                ui.mark(type='line', x='=month', y='=price', curve='step-before', y_min=0)
            ])
        )
# --------------        
        await q.page.save()

        # Run score asynchronously
        new_dir = '/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1'
        os.chdir(new_dir)
        number = await q.run(run_scoring_pipeline)
        percent = number/30*100

        # Update the content of the 'example2' card with the result
        q.page['example2'].content = f'Number of people in the frame is {number}'
        q.page['hello'] = ui.markdown_card(
        box='6 2 5 1',
        title='Capacity',
        content=f'Capacity is at {percent}% full!!'
    )
#----------------------------------
        import csv

        # Specify the CSV file name
        csv_file_name = 'Mall_Customers.csv'

        # Specify the path to the CSV file
        csv_file_path = '/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/Mall_Customers.csv'  # Update this with the correct path

        rows_per_page = 10

        if not q.app.initialized:
            # Allow downloading all data since no filters/search/sort is allowed.
            # For multi-user apps, the tmp file name should be unique for each user, not hardcoded.
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
                for i in range(1, 101):
                    csv_writer.writerow([i, 'Male' if i % 2 == 1 else 'Female', i % 50 + 15, i % 50, i % 100])

            q.app.data_download, = await q.site.upload([csv_file_name])
            q.app.initialized = True

        if not q.client.initialized:
            q.page['meta'] = ui.meta_card(box='')
            q.page['form'] = ui.form_card(box='1 6 5 6', items=[
                ui.table(
                    name='table',
                    columns=[ui.table_column(name=col, label=col, link=False) for col in ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']],
                    rows=[ui.table_row(name=str(i), cells=[str(i), 'Male' if i % 2 == 1 else 'Female', str(i % 50 + 15), str(i % 50), str(i % 100)]) for i in range(1, rows_per_page + 1)],
                    pagination=ui.table_pagination(total_rows=100, rows_per_page=rows_per_page),
                    downloadable=True,
                    events=['page_change', 'download']
                )
            ])
            q.client.initialized = True

        if q.events.table:
            if q.events.table.download:
                q.page['meta'].script = ui.inline_script(f'window.open("{q.app.data_download}")')
            if q.events.table.page_change:
                offset = q.events.table.page_change.get('offset', 0)
                new_rows = [[str(i), 'Male' if i % 2 == 1 else 'Female', str(i % 50 + 15), str(i % 50), str(i % 100)] for i in range(offset + 1, offset + 1 + rows_per_page)]
                q.page['form'].table.rows = [ui.table_row(name=str(i), cells=row) for i, row in enumerate(new_rows, start=offset + 1)]
#----------------------------------
        await q.page.save()
    else:
        # Show the file upload form
        q.page['example'] = ui.form_card(
            box='1 1 4 4',
            items=[
                ui.file_upload(name='file_upload', label='Select a .jpg file to upload', compact=True, multiple=False,
                               file_extensions=['jpg'], max_file_size=5, max_size=15),
                ui.button(name='submit', label='Submit', primary=True)
            ]
        )

    await q.page.save()

if __name__ == '__main__':
    main('/', port=10101)

