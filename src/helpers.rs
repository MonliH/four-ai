use std::error::Error;
use std::fs;
use std::path;

pub fn get_max_generation(file_path: &path::Path) -> Result<Option<fs::DirEntry>, Box<dyn Error>> {
    Ok(
        fs::read_dir(file_path.parent().unwrap_or(path::Path::new("./")))?
            .map(|file| file.expect(concat!(RED!(), "File failed", RESET!())))
            .max_by_key(|val| {
                val.file_name()
                    .to_str()?
                    .split("_")
                    .last()?
                    .parse::<usize>()
                    .ok()
            }),
    )
}
