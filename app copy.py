import os
from h2o_wave import main, app, Q, ui, AsyncSite
import shutil  ##

# Set the path to the images folder
IMAGES_FOLDER = '/home/ladmin/inspiredemo/waveserver/venvwave/scoring_puzzling-tarsier_fold1/images'

# Create the images folder if it doesn't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Define the Wave app
@app('/')
async def serve(q: Q):
    if 'file_upload' in q.args:
        # Save the uploaded image to the images folder
        uploaded_file = q.args.file_upload[0]
#-----------------
        file_to_copy = '/home/ladmin/inspiredemo/waveserver/venvwave/data/' + uploaded_file[2:]
        shutil.copy(file_to_copy, IMAGES_FOLDER)
        #path_to_copy = '/home/ladmin/inspiredemo/waveserver/venvwave/data/' + uploaded_file[2:40]
#--------------------
        #image_path = os.path.join(IMAGES_FOLDER, uploaded_file)
        # Here, uploaded_file is a string representing the file name
        # Now, you can display the confirmation message with the filename
        q.page['example'] = ui.form_card(box='1 1 5 8', items=[
            ui.text(f'Uploaded image saved with the original filename: {uploaded_file}'),
            ui.button(name='show_upload', label='Back', primary=True),
            ui.image(title=file_to_copy, path=uploaded_file)
        ])

        count = 0
        # Add a Markdown card named `hello` to the page.
        q.page['example2'] = ui.markdown_card(
            box='6 1 5 8',
            title='Prediction',
            content='Number  of people in the frame is ' + str(count),
        )

    else:
        # Show the file upload form
        q.page['example'] = ui.form_card(
            box='1 1 4 7',
            items=[
                ui.file_upload(name='file_upload', label='Select a .jpg file to upload', compact=True, multiple=False,
                               file_extensions=['jpg'], max_file_size=5, max_size=15),
                ui.button(name='submit', label='Submit', primary=True)
            ]
        )

    await q.page.save()

if __name__ == '__main__':
    main('/', port=10101)

