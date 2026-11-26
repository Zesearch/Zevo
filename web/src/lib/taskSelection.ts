/** Resolve a selected resource without falling back to another owner's same-name task. */
export function resolveLaunchTask<T extends { id?: string; name: string }>(
  tasks: T[], name: string, id = "",
): T | undefined {
  const matches = tasks.filter(task => task.name === name.trim());
  if (id) return matches.find(task => task.id === id);
  return matches.length === 1 ? matches[0] : undefined;
}
