from os.path import isfile


def args_save_to_txt(save_file_path, args):
    from datetime import date
    with open(save_file_path, 'w') as f:
        f.write(f"Date: {date.today()}\n\n")
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
        f.close()

    assert isfile(save_file_path)


if __name__ == '__main__':
    pass
