import { useEffect, useRef, useState } from "react";
import { getRuns } from "../lib/api";

export function useRuns() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const ctrlRef = useRef(null);

  const reload = async () => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true);
    setErr(null);
    try {
      const data = await getRuns(ctrl.signal);
      setRuns(Array.isArray(data) ? data : []);
    } catch (e) {
      setErr(e.message || "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
    return () => ctrlRef.current?.abort();
  }, []);
  return { runs, loading, err, reload, setRuns };
}
