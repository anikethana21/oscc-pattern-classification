modules = ['tifffile', 'openslide', 'cv2', 'PIL', 'imageio']
installed = []
for m in modules:
    try:
        __import__(m)
        installed.append(m)
    except Exception as e:
        pass
print("Installed WSI modules:", installed)

