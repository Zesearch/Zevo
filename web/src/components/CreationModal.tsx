import type { ReactNode } from "react";
import { Modal } from "./Modal";
import { ClearFieldsContext } from "./ClearableField";
import "./launchForm.css";
import "./creationForm.css";

/** Task and file creation share the spacious Launch Run form treatment. */
export function CreationModal({ open, title, heading, description, onClose, children, wide = false }: {
  open: boolean;
  title: string;
  heading: string;
  description: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  return <ClearFieldsContext.Provider value={true}><Modal fullscreen open={open} title={title} onClose={onClose}>
    <div className="launch-experience creation-experience mx-auto w-full max-w-[1400px] pb-10 md:w-[94%]">
      <div className={`mx-auto ${wide ? "max-w-[1280px]" : "max-w-[1000px]"}`}>
        <header className="creation-header pt-4 md:pt-8">
          <h3 className="creation-title font-display text-ink">{heading}</h3>
          <p className="creation-intro max-w-2xl">{description}</p>
        </header>
        {children}
      </div>
    </div>
  </Modal></ClearFieldsContext.Provider>;
}
