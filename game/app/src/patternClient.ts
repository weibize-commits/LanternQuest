export interface GeneratedPatternResponse {
  asset_id: string;
  image_data_url: string;
  model: string;
  created_at: string;
}

export class PatternClientError extends Error {
  constructor(public code: string, message: string) {
    super(message);
    this.name = "PatternClientError";
  }
}

export async function generatePattern(prompt: string): Promise<GeneratedPatternResponse> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), 150_000);
  try {
    const response = await fetch("/api/pattern", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt.trim() }),
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok) {
      throw new PatternClientError(
        typeof payload.code === "string" ? payload.code : `http_${response.status}`,
        typeof payload.message === "string" ? payload.message : "图案生成暂时不可用",
      );
    }
    if (
      typeof payload.asset_id !== "string" ||
      typeof payload.image_data_url !== "string" ||
      !payload.image_data_url.startsWith("data:image/") ||
      typeof payload.model !== "string" ||
      typeof payload.created_at !== "string"
    ) {
      throw new PatternClientError("invalid_response", "图案服务返回了无效结果");
    }
    return payload as unknown as GeneratedPatternResponse;
  } catch (error) {
    if (error instanceof PatternClientError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new PatternClientError("timeout", "生成时间过长，请稍后重试");
    }
    throw new PatternClientError("network", "无法连接图案生成服务");
  } finally {
    globalThis.clearTimeout(timeout);
  }
}
