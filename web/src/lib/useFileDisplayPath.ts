import useSWR from "swr";
import type { FileSetDTO } from "./api";
import { displayFilePath } from "./format";

/** Resolve labels from the same catalogue as Files without rewriting paths. */
export function useFileDisplayPath() {
  const { data: files = [] } = useSWR<FileSetDTO[]>("/api/files");
  return (path: string) => displayFilePath(path, files);
}
