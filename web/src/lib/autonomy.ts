/** Classify any combination of the three user-pinned training decisions. */
export function levelOf(dataset: string, baseModel: string, trainingMethod: string): string {
  const pinned = [dataset, baseModel, trainingMethod].filter((value) => value.trim()).length;
  return `L${4 - pinned}`;
}
