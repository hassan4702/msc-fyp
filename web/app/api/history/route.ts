import { and, asc, desc, eq } from "drizzle-orm";
import { headers } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { conversation, message } from "@/lib/schema";

async function currentUserId() {
  const session = await auth.api.getSession({ headers: await headers() });
  return session?.user?.id ?? null;
}

// GET /api/history            -> list the user's conversations
// GET /api/history?id=<id>    -> messages of one conversation (must own it)
export async function GET(req: NextRequest) {
  const uid = await currentUserId();
  if (!uid) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const id = req.nextUrl.searchParams.get("id");
  if (id) {
    const [conv] = await db
      .select()
      .from(conversation)
      .where(and(eq(conversation.id, id), eq(conversation.userId, uid)));
    if (!conv) return NextResponse.json({ error: "not found" }, { status: 404 });
    const messages = await db
      .select({ role: message.role, content: message.content })
      .from(message)
      .where(eq(message.conversationId, id))
      .orderBy(asc(message.createdAt));
    return NextResponse.json({ id, title: conv.title, messages });
  }

  const conversations = await db
    .select({ id: conversation.id, title: conversation.title })
    .from(conversation)
    .where(eq(conversation.userId, uid))
    .orderBy(desc(conversation.createdAt));
  return NextResponse.json({ conversations });
}

// POST /api/history  { conversationId?, turn: { userText, botText, emotion } }
// creates the conversation on first turn; returns the (possibly new) conversation id
export async function POST(req: NextRequest) {
  const uid = await currentUserId();
  if (!uid) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const { conversationId, turn } = await req.json();
  if (!turn?.userText || !turn?.botText) {
    return NextResponse.json({ error: "bad request" }, { status: 400 });
  }

  let convId: string = conversationId;
  let title = "New chat";
  if (convId) {
    const [conv] = await db
      .select({ title: conversation.title })
      .from(conversation)
      .where(and(eq(conversation.id, convId), eq(conversation.userId, uid)));
    if (!conv) return NextResponse.json({ error: "not found" }, { status: 404 });
    title = conv.title;
  } else {
    convId = crypto.randomUUID();
    title = turn.userText.slice(0, 60);
    await db.insert(conversation).values({ id: convId, userId: uid, title });
  }

  await db.insert(message).values([
    { id: crypto.randomUUID(), conversationId: convId, role: "user", content: turn.userText, emotion: turn.emotion ?? null },
    { id: crypto.randomUUID(), conversationId: convId, role: "bot", content: turn.botText },
  ]);

  return NextResponse.json({ conversationId: convId, title });
}
